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
#include "gtest_utils_ai.hpp"

using namespace ai_gtests;
#include <iostream>
#include <memory>
#include <cstdlib>

/** @brief AI Test class for comprehensive ZenDNNL matmul testing */
class TestMatmulAI : public ::testing::TestWithParam<MatmulParamsAI> {
  // -----------------------------------------------------------------------------
  // CreatedTensors
  //
  // Helper struct for storing all tensors created for a matmul test.
  // Used to pass input, weights, bias, output, reference output, and binary post-op tensors
  // between test flows. All test flows use this struct for consistency.
  // -----------------------------------------------------------------------------
  struct CreatedTensors {
    tensor_t input;
    tensor_t weights;
    tensor_t bias;
    tensor_t output;
    tensor_t reference_output; // optional, may be empty
    std::vector<std::pair<std::string, tensor_t>> binary_post_op_tensors;
  };

  // -----------------------------------------------------------------------------
  // create_test_tensors
  //
  // Centralized tensor creation helper for all matmul test flows.
  // Creates input, weights, bias, output, reference output, and binary post-op tensors
  // with standardized category-based naming. Used by all test categories to ensure
  // consistent tensor setup and naming. Returns a CreatedTensors struct.
  //
  // Parameters:
  //   params - Matmul test parameters
  //   input_dtype - Data type for input tensor
  //   weight_dtype - Data type for weights tensor
  //   output_dtype - Data type for output tensor
  //   use_boundary - If true, create boundary tensors; else, uniform tensors
  //   create_reference_output - If true, create reference output tensor
  //   name_suffix - Suffix for tensor names (category-based)
  // Returns:
  //   Struct containing all created tensors
  // -----------------------------------------------------------------------------
  CreatedTensors create_test_tensors(const MatmulParamsAI &params,
                                     data_type_t input_dtype,
                                     data_type_t weight_dtype,
                                     data_type_t output_dtype,
                                     bool use_boundary = false,
                                     bool create_reference_output = false,
                                     const std::string &name_suffix = "") {
    CreatedTensors tensors;
    auto tensor_creator = use_boundary ? &AITensorFactory::create_boundary_tensor
                          : &AITensorFactory::create_uniform_tensor;

    std::string input_name = "input" + name_suffix;
    std::string weights_name = "weights" + name_suffix;
    std::string bias_name = "bias" + name_suffix;
    std::string output_name = "output_zero" + name_suffix;
    std::string ref_output_name = "ref_output_zero" + name_suffix;

    tensors.input = tensor_creator({params.m, params.k}, input_dtype, input_name);
    tensors.weights = tensor_creator({params.k, params.n}, weight_dtype,
                                     weights_name);
    tensors.bias = tensor_creator({1, params.n}, output_dtype, bias_name);
    tensors.output = AITensorFactory::create_zero_tensor({params.m, params.n},
                     output_dtype, output_name);
    if (create_reference_output)
      tensors.reference_output = AITensorFactory::create_zero_tensor({params.m, params.n},
                                 output_dtype, ref_output_name);

    int add_count = 0, mul_count = 0;
    for (const auto &post_op_type : params.post_op_config.post_ops) {
      if (post_op_type == post_op_type_t::binary_add) {
        std::string name = "binary_add_tensor_" + std::to_string(
                             add_count) + name_suffix;
        auto tensor = tensor_creator({params.m, params.n}, output_dtype, name);
        tensors.binary_post_op_tensors.emplace_back(name, tensor);
        ++add_count;
      }
      else if (post_op_type == post_op_type_t::binary_mul) {
        std::string name = "binary_mul_tensor_" + std::to_string(
                             mul_count) + name_suffix;
        auto tensor = tensor_creator({params.m, params.n}, output_dtype, name);
        tensors.binary_post_op_tensors.emplace_back(name, tensor);
        ++mul_count;
      }
      else {
        // For any other post-op, create an empty tensor and add to the list
        std::string name = "empty_post_op_tensor_" + name_suffix;
        tensor_t empty_tensor;
        tensors.binary_post_op_tensors.emplace_back(name, empty_tensor);
      }
    }
    return tensors;
  }
 protected:
  // -----------------------------------------------------------------------------
  // SetUp
  //
  // Test fixture setup for each test case.
  // Initializes parameter generator, clears previous test state, and logs test parameters.
  // Called automatically before each test runs.
  // -----------------------------------------------------------------------------
  virtual void SetUp() override {
    // Add speific setup code here if needed
  }

  // -----------------------------------------------------------------------------
  // TearDown
  //
  // Test fixture teardown for each test case.
  // Cleans up after each test. Called automatically after each test runs.
  // -----------------------------------------------------------------------------
  virtual void TearDown() override {
    // Cleanup after each test
  }

  // -----------------------------------------------------------------------------
  // run_reference_kernel_test
  //
  // Runs the reference kernel only for exhaustive reference kernel tests.
  // Validates output and logs results. Used for tests where only the reference
  // implementation is required (no ZenDNNL kernel comparison).
  //
  // Parameters:
  //   params - Matmul test parameters
  // -----------------------------------------------------------------------------
  void run_reference_kernel_test(const MatmulParamsAI &params) {
    ASSERT_TRUE(AITestUtils::validate_dimensions(params.m, params.n, params.k))
        << "Invalid dimensions for reference kernel test";

    auto input_dtype = AITestUtils::get_input_dtype(params.data_types);
    auto weight_dtype = AITestUtils::get_weight_dtype(params.data_types);
    auto output_dtype = AITestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, input_dtype, weight_dtype,
                                       output_dtype, false, false, "_reference");

    // Convert binary_post_op_tensors to vector<tensor_t> for reference kernel
    std::vector<tensor_t> ref_binary_post_op_tensors;
    for (const auto &pair : tensors.binary_post_op_tensors) {
      ref_binary_post_op_tensors.push_back(pair.second);
    }

    // Run reference kernel only
    status_t ref_status = AITestUtils::run_reference_matmul(
                            tensors.input, tensors.weights, tensors.bias, tensors.output,
                            params.post_op_config, ref_binary_post_op_tensors);

    if (AITestUtils::is_reference_implementation_supported(input_dtype,
        weight_dtype, output_dtype, params.post_op_config.post_ops)) {
      EXPECT_EQ(ref_status, status_t::success)
          << "Reference kernel must succeed for supported dtype/post-op combinations";
    }
    else {
      EXPECT_NE(ref_status, status_t::success)
          << "Reference kernel should fail for non-supported post-ops";
    }
  }
  // -----------------------------------------------------------------------------
  // get_accuracy_tolerance
  //
  // Returns the accuracy tolerance for the given output data type.
  // Used to determine the threshold for output comparison in accuracy tests.
  //
  // Parameters:
  //   output_dtype - Output tensor data type
  // Returns:
  //   Tolerance value for accuracy comparison
  // -----------------------------------------------------------------------------
  float get_accuracy_tolerance(data_type_t output_dtype) const {
    switch (output_dtype) {
    case data_type_t::f32:
      return AI_MATMUL_TOLERANCE_F32;
    case data_type_t::bf16:
      return AI_MATMUL_TOLERANCE_BF16;
    case data_type_t::s8:
      return AI_MATMUL_TOLERANCE_S8;
    default:
      return AI_MATMUL_TOLERANCE_DEFAULT;
    }
  }

  float get_relative_tolerance(data_type_t output_dtype) const {
    switch (output_dtype) {
    case data_type_t::f32:
      return AI_MATMUL_REL_TOLERANCE_F32;
    case data_type_t::bf16:
      return AI_MATMUL_REL_TOLERANCE_BF16;
    case data_type_t::s8:
      return AI_MATMUL_REL_TOLERANCE_S8;
    default:
      return AI_MATMUL_REL_TOLERANCE_DEFAULT;
    }
  }

// -----------------------------------------------------------------------------
// get_epsilon_value
//
// Returns the epsilon value for the given data type, used for floating-point comparison.
// Uses AI_MATMUL_EPSILON_F32, AI_MATMUL_EPSILON_BF16, etc.
// -----------------------------------------------------------------------------
  float get_epsilon_value(data_type_t dtype) {
    switch (dtype) {
    case data_type_t::f32:
      return AI_MATMUL_EPSILON_F32;
    case data_type_t::bf16:
      return AI_MATMUL_EPSILON_BF16;
    case data_type_t::s8:
      return AI_MATMUL_EPSILON_S8;
    default:
      return AI_MATMUL_EPSILON_DEFAULT;
    }
  }

  // -----------------------------------------------------------------------------
  // run_accuracy_test
  //
  // Runs accuracy test by comparing ZenDNNL kernel output against reference implementation.
  // Handles both reference-supported and non-reference-supported cases.
  // Used for validating correctness of ZenDNNL kernel output.
  //
  // Parameters:
  //   params - Matmul test parameters
  // -----------------------------------------------------------------------------
  void run_accuracy_test(const MatmulParamsAI &params) {
    ASSERT_TRUE(AITestUtils::validate_dimensions(params.m, params.n, params.k))
        << "Invalid dimensions for accuracy test";

    auto input_dtype = ai_gtests::AITestUtils::get_input_dtype(params.data_types);
    auto weight_dtype = ai_gtests::AITestUtils::get_weight_dtype(params.data_types);
    auto output_dtype = ai_gtests::AITestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, input_dtype, weight_dtype,
                                       output_dtype, false, true, "_accuracy");
    if (!tensors.input.get_raw_handle_const()) {
      AITestUtils::debug_print("[AI_DEBUG][FATAL] Input tensor data pointer is null!");
      std::abort();
    }
    if (!tensors.weights.get_raw_handle_const()) {
      AITestUtils::debug_print("[AI_DEBUG][FATAL] Weights tensor data pointer is null!");
      std::abort();
    }
    if (!tensors.output.get_raw_handle_const()) {
      AITestUtils::debug_print("[AI_DEBUG][FATAL] Output tensor data pointer is null!");
      std::abort();
    }
    if (!tensors.reference_output.get_raw_handle_const()) {
      AITestUtils::debug_print("[AI_DEBUG][FATAL] Reference output tensor data pointer is null!");
      std::abort();
    }

    //AITestUtils::log_tensor_info(tensors.input, "input");
    //AITestUtils::log_tensor_info(tensors.weights, "weights");
    //AITestUtils::log_tensor_info(tensors.output, "output");

    AITestUtils::debug_print("[AI_DEBUG] About to run ZenDNNL kernel...");
    bool aocl_supported = AITestUtils::is_aocl_kernel_supported(input_dtype,
                          weight_dtype, output_dtype, params.post_op_config.post_ops);
    status_t test_status = run_matmul_test(tensors.input, tensors.weights,
                                           tensors.bias, tensors.output, params, tensors.binary_post_op_tensors);
    AITestUtils::debug_print("[AI_DEBUG] ZenDNNL kernel finished.");
    if (aocl_supported) {
      EXPECT_EQ(test_status, status_t::success)
          << "ZenDNNL AOCL kernel must succeed for supported data types";
    }
    else {
      EXPECT_NE(test_status, status_t::success)
          << "ZenDNNL AOCL kernel should fail for unsupported data types";
    }

    // **ACCURACY TEST LOGIC BASED ON REFERENCE IMPLEMENTATION SUPPORT**
    if (AITestUtils::is_reference_implementation_supported(input_dtype,
        weight_dtype, output_dtype, params.post_op_config.post_ops)) {
      // **REFERENCE SUPPORTED: ZenDNNL must succeed AND match reference implementation**
      if (test_status == status_t::success) {
        // Run reference implementation for accuracy comparison
        AITestUtils::debug_print("[AI_DEBUG] About to run reference implementation...");
        std::vector<tensor_t> ref_binary_post_op_tensors;
        for (const auto &pair : tensors.binary_post_op_tensors) {
          ref_binary_post_op_tensors.push_back(pair.second);
        }
        status_t ref_status = AITestUtils::run_reference_matmul(
                                tensors.input, tensors.weights, tensors.bias,
                                tensors.reference_output,
                                params.post_op_config,
                                ref_binary_post_op_tensors);
        AITestUtils::debug_print("[AI_DEBUG] Reference implementation finished.");
        EXPECT_EQ(ref_status, status_t::success)
            << "Reference implementation must succeed for supported data types";
        if (ref_status == status_t::success) {
          // Compare ZenDNNL output with reference implementation
          float rel_tolerance = get_relative_tolerance(output_dtype);
          float epsilon = get_epsilon_value(output_dtype);
          bool comparison_result = AITestUtils::compare_sampled_tensors_matmul(
                                     tensors.output, tensors.reference_output, params.k, rel_tolerance, epsilon);
          EXPECT_TRUE(comparison_result)
              << "ZenDNNL output must match reference within abs_bound + rtol*|ref|, where abs_bound (epsilon-based) = "
              << epsilon
              << ", rel tolerance: " << rel_tolerance;
        }
      }
    }
    else {
      // **REFERENCE NOT SUPPORTED: Skip reference comparison, validate kernel behavior**
      if (params.expect_success) {
        if (test_status == status_t::success) {
          // Validate that output tensor is reasonable
          EXPECT_TRUE(validate_output_tensor(tensors.output, params))
              << "ZenDNNL output tensor validation failed";
        }
      }
      else {
        EXPECT_NE(test_status, status_t::success)
            << "Test succeeded when failure was expected";
      }
    }
  }

  // -----------------------------------------------------------------------------
  // run_boundary_test
  //
  // Runs boundary condition test for matmul (e.g., extreme values, edge dimensions).
  // Validates output for numerical stability and correctness. Used to test kernel
  // behavior under boundary conditions and extreme values.
  //
  // Parameters:
  //   params - Matmul test parameters
  // -----------------------------------------------------------------------------
  void run_boundary_test(const MatmulParamsAI &params) {
    // **REAL BOUNDARY CONDITION TESTING**
    auto input_dtype = AITestUtils::get_input_dtype(params.data_types);
    auto weight_dtype = AITestUtils::get_weight_dtype(params.data_types);
    auto output_dtype = AITestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, input_dtype, weight_dtype,
                                       output_dtype, true, false, "_boundary");

    status_t test_status = run_matmul_test(tensors.input, tensors.weights,
                                           tensors.bias, tensors.output, params, tensors.binary_post_op_tensors);

    if (params.expect_success) {
      EXPECT_EQ(test_status, status_t::success)
          << "Boundary test failed when success was expected for " << params.test_name;

      // **BOUNDARY-SPECIFIC VALIDATION**
      // Check for numerical stability and expected behavior
      EXPECT_TRUE(validate_boundary_output(tensors.output, params))
          << "Boundary output validation failed for " << params.test_name;

      // Check for precision-related issues
      EXPECT_TRUE(validate_numerical_stability(tensors, params))
          << "Numerical stability check failed for " << params.test_name;

    }
    else {
      EXPECT_NE(test_status, status_t::success)
          << "Boundary test succeeded when failure was expected for " << params.test_name;
    }

    std::cout << "[AI_BOUNDARY] " << params.test_name << " completed with status: "
              << static_cast<int>(test_status) << std::endl;
  }

  /**
   * @brief Runs edge case test for matmul (e.g., unusual but valid shapes).
   *        Validates kernel success/failure as expected. Used to test kernel robustness
   *        for edge-case shapes and parameters.
   *
   * @param params Matmul test parameters
   * @note Used to verify kernel correctness for rare or irregular input shapes.
   */
  void run_edge_case_test(const MatmulParamsAI &params) {
    auto input_dtype = AITestUtils::get_input_dtype(params.data_types);
    auto weight_dtype = AITestUtils::get_weight_dtype(params.data_types);
    auto output_dtype = AITestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, input_dtype, weight_dtype,
                                       output_dtype, false, false, "_edge");

    status_t test_status = run_matmul_test(tensors.input, tensors.weights,
                                           tensors.bias, tensors.output, params, tensors.binary_post_op_tensors);

    if (params.expect_success) {
      EXPECT_EQ(test_status, status_t::success)
          << "Edge case test failed when success was expected";
    }
    else {
      EXPECT_NE(test_status, status_t::success)
          << "Edge case test succeeded when failure was expected";
    }
  }

  /**
   * @brief Runs invalid case test for matmul (e.g., invalid shapes, missing tensors).
   *        Expects kernel to fail for all invalid scenarios. Used to verify kernel error
   *        handling and robustness against invalid inputs.
   *
   * @param params Matmul test parameters
   * @note Used to ensure error handling and negative code paths are exercised.
   */
  void run_invalid_test(const MatmulParamsAI &params) {
    // For invalid tests, we expect failures
    EXPECT_FALSE(params.expect_success)
        << "Invalid test should have expect_success = false";

    // **RUN ACTUAL MATMUL KERNEL WITH INVALID DIMENSIONS**
    // Do not skip - test the actual kernel behavior with invalid inputs
    status_t test_status = run_invalid_matmul_test(params);

    // Invalid tests should fail
    EXPECT_NE(test_status, status_t::success)
        << "Invalid test should fail but succeeded: " << params.test_name;
  }



 private:
  // -----------------------------------------------------------------------------
  // run_invalid_matmul_test
  //
  // Simulates all invalid matmul scenarios and verifies kernel failure.
  // Handles specific cases (null weights, shape mismatch, missing tensors, etc.).
  // Used internally by run_invalid_test to cover all negative code paths.
  //
  // Parameters:
  //   params - Matmul test parameters
  // Returns:
  //   Failure status if kernel behaves correctly
  // -----------------------------------------------------------------------------
  status_t run_invalid_matmul_test(const MatmulParamsAI &params) {
    // Explicit scenario-based branching for all negative/invalid code paths
    // Use test_name substring matching for case_type
    std::string case_type = params.test_name;

    auto input_dtype = AITestUtils::get_input_dtype(params.data_types);
    auto weight_dtype = AITestUtils::get_weight_dtype(params.data_types);
    auto output_dtype = AITestUtils::get_output_dtype(params.data_types);
    std::vector<uint64_t> input_dims = {params.m, params.k};
    std::vector<uint64_t> weight_dims = {params.k, params.n};
    std::vector<uint64_t> bias_dims = {1, params.n};
    std::vector<uint64_t> output_dims = {params.m, params.n};

    // 1. Weights tensor is null
    if (case_type.find("weights_null") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        matmul_context_t matmul_context = matmul_context_t().set_param("bias",
                                          bias).create();
        if (!matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 2. Weights not 2D
    if (case_type.find("weights_not_2d") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor({params.k, params.n, 2},
                         weight_dtype, "weights3d_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 3. Bias size mismatch
    if (case_type.find("bias_size_mismatch") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor({1, params.n + 1},
                      output_dtype, "bias_bad_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 4. Not binding input tensor
    if (case_type.find("missing_input") != std::string::npos) {
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 5. Not binding output tensor
    if (case_type.find("missing_output") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input", input).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 6. Input or output not 2D
    if (case_type.find("input_not_2d") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor({params.m, params.k, 2},
                       input_dtype, "input3d_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }
    if (case_type.find("output_not_2d") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor({params.m, params.n, 2},
                        output_dtype, "output3d_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 7. Tensor order set to invalid value
    if (case_type.find("tensor_order_invalid") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        input.set_order("ba"); // Simulate invalid order
      }
      catch (...) {}
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 8. Forced kernel to unsupported value
    if (case_type.find("forced_kernel_unsupported") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        tensor_t forced_kernel_tensor; // must be lvalue
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).set_param("forced_kernel",
                                  forced_kernel_tensor).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }
    // 8b. Forced kernel set to unknown string
    if (case_type.find("forced_kernel_unknown") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        matmul_context_t matmul_context = matmul_context_t().set_param("weights",
                                          weights).set_param("bias", bias).create();
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).set_forced_kernel("foobar_kernel").create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }
    // 8c. Forced kernel set to empty string
    if (case_type.find("forced_kernel_empty") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        matmul_context_t matmul_context = matmul_context_t().set_param("weights",
                                          weights).set_param("bias", bias).create();
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).set_forced_kernel("").create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }
    // 8d. Unknown/unsupported post-op type
    if (case_type.find("unknown_post_op") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        post_op_t bad_post_op(static_cast<post_op_type_t>(999));
        matmul_context_t matmul_context = matmul_context_t().set_param("weights",
                                          weights).set_param("bias", bias).create();
        matmul_context = matmul_context.set_post_op(bad_post_op).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }
    // 8e. Mixed valid/invalid post-ops
    if (case_type.find("mixed_post_op") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        post_op_t relu_post_op(post_op_type_t::relu);
        post_op_t bad_post_op(static_cast<post_op_type_t>(999));
        matmul_context_t matmul_context = matmul_context_t().set_param("weights",
                                          weights).set_param("bias", bias).create();
        matmul_context = matmul_context.set_post_op(relu_post_op).set_post_op(
                           bad_post_op).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 9. Malformed or missing binary post-op tensor
    if (case_type.find("binary_post_op_missing") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        post_op_t binary_add_post_op(post_op_type_t::binary_add);
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).set_post_op(binary_add_post_op).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        // Do NOT bind the required binary post-op tensor
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 10. Dimension mismatch (e.g., input and weights shapes don't align)
    if (case_type.find("dim_mismatch") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor({params.m, params.k + 1},
                       input_dtype, "input_bad_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // 11. Unimplemented kernel for unsupported dtypes
    if (case_type.find("unsupported_dtype") != std::string::npos) {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims,
                       data_type_t::s4, "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         data_type_t::s4, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims,
                      data_type_t::s4, "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims,
                        data_type_t::s4, "output_zero");
      try {
        auto matmul_context = matmul_context_t().set_param("weights",
                              weights).set_param("bias", bias).create();
        if (! matmul_context.check()) {
          return status_t::failure;
        }
        matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                              matmul_context).create();
        matmul_operator.set_input("matmul_input",
                                  input).set_output("matmul_output", output).execute();
        return status_t::failure; // Invalid test should always fail
      }
      catch (...) {
        return status_t::failure;
      }
    }

    // Default fallback: try to run with whatever invalid dims are present
    try {
      tensor_t input = AITensorFactory::create_uniform_tensor(input_dims, input_dtype,
                       "input_accuracy");
      tensor_t weights = AITensorFactory::create_uniform_tensor(weight_dims,
                         weight_dtype, "weights_accuracy");
      tensor_t bias = AITensorFactory::create_uniform_tensor(bias_dims, output_dtype,
                      "bias_accuracy");
      tensor_t output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                        "output_zero");
      matmul_context_t matmul_context = matmul_context_t().set_param("weights",
                                        weights).set_param("bias", bias).create();
      if (!matmul_context.check()) {
        return status_t::failure;
      }
      matmul_operator_t matmul_operator = matmul_operator_t().set_context(
                                            matmul_context).create();
      matmul_operator.set_input("matmul_input",
                                input).set_output("matmul_output", output).execute();
      return status_t::failure; // Invalid test should always fail
    }
    catch (...) {
      return status_t::failure;
    }
  }

  // -----------------------------------------------------------------------------
  // run_matmul_test
  //
  // Core matmul test execution for ZenDNNL kernel.
  // Runs ZenDNNL matmul kernel with provided tensors and parameters, including post-ops
  // and binary post-op tensor binding. Used by all test flows except reference-only tests.
  //
  // Parameters:
  //   input - Input tensor
  //   weights - Weights tensor
  //   bias - Bias tensor
  //   output - Output tensor
  //   params - Matmul test parameters
  //   binary_post_op_tensors - Binary post-op tensors (name, tensor pairs)
  // Returns:
  //   Success or failure status
  // -----------------------------------------------------------------------------
  status_t run_matmul_test(tensor_t &input, tensor_t &weights,
                           tensor_t &bias, tensor_t &output,
                           const MatmulParamsAI &params,
                           std::vector<std::pair<std::string, tensor_t>> &binary_post_op_tensors) {
    try {
      matmul_context_t matmul_context = matmul_context_t()
                                        .set_param("weights", weights)
                                        .set_param("bias", bias);
      for (const auto &post_op_type : params.post_op_config.post_ops) {
        post_op_t post_op{post_op_type};
        matmul_context = matmul_context.set_post_op(post_op);
      }
      matmul_context = matmul_context.create();
      if (!matmul_context.check()) {
        std::cout << "[AI_TEST] Context creation failed for " << params.test_name <<
                  std::endl;
        return status_t::failure;
      }
      auto matmul_operator = matmul_operator_t()
                             .set_name(AITestUtils::generate_unique_name("matmul_ai_op"))
                             .set_context(matmul_context)
                             .create();
      if (matmul_operator.is_bad_object()) {
        std::cout << "[AI_TEST] Operator creation failed for " << params.test_name <<
                  std::endl;
        return status_t::failure;
      }
      // Bind extra tensors for binary post-ops using context-derived names
      size_t binary_tensor_idx = 0;
      for (size_t i = 0; i < params.post_op_config.post_ops.size(); ++i) {
        auto post_op_type = params.post_op_config.post_ops[i];
        if ((post_op_type == post_op_type_t::binary_add ||
             post_op_type == post_op_type_t::binary_mul)
            && binary_tensor_idx < binary_post_op_tensors.size()) {
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
            tensor_name = binary_post_op_tensors[binary_tensor_idx].first; // fallback
          }
          matmul_operator = matmul_operator.set_input(tensor_name,
                            binary_post_op_tensors[binary_tensor_idx].second);
          binary_tensor_idx++;
        }
      }
      auto status = matmul_operator
                    .set_input("matmul_input", input)
                    .set_output("matmul_output", output)
                    .execute();

      return status;

    }
    catch (const std::exception &e) {
      std::cout << "[AI_TEST] Exception in " << params.test_name << ": " << e.what()
                << std::endl;
      return status_t::failure;
    }
    catch (...) {
      std::cout << "[AI_TEST] Unknown exception in " << params.test_name << std::endl;
      return status_t::failure;
    }
  }

  // -----------------------------------------------------------------------------
  // validate_output_tensor
  //
  // Validates basic properties of output tensor (dimensions, data type, nonzero values).
  // Used in accuracy and boundary tests to check output tensor validity.
  //
  // Parameters:
  //   output - Output tensor
  //   params - Matmul test parameters
  // Returns:
  //   true if output tensor is valid, false otherwise
  // -----------------------------------------------------------------------------
  bool validate_output_tensor(const tensor_t &output,
                              const MatmulParamsAI &params) {
    // Check dimensions
    auto expected_dims = std::vector<uint64_t> {params.m, params.n};
    if (output.get_size() != expected_dims) {
      return false;
    }

    // Check data type
    auto expected_dtype = AITestUtils::get_output_dtype(params.data_types);
    if (output.get_data_type() != expected_dtype) {
      return false;
    }

    // Check that output is not all zeros (basic sanity check)
    if (output.get_nelem() > 0) {
      auto dtype = output.get_data_type();
      bool has_non_zero = false;
      auto sample_indices = AITestUtils::get_sample_indices(output.get_nelem(), 100);
      if (dtype == data_type_t::f32) {
        const float *data = static_cast<const float *>(output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          if (std::abs(data[idx]) > 1e-7f) {
            has_non_zero = true;
            break;
          }
        }
      }
      else if (dtype == data_type_t::bf16) {
        const bfloat16_t *data = static_cast<const bfloat16_t *>
                                 (output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          if (std::abs(static_cast<float>(data[idx])) > 1e-4f) {
            has_non_zero = true;
            break;
          }
        }
      }
      else if (dtype == data_type_t::s8 || dtype == data_type_t::s4) {
        const int8_t *data = static_cast<const int8_t *>(output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          if (data[idx] != 0) {
            has_non_zero = true;
            break;
          }
        }
      }
      else {
        // For other types, fallback: check raw bytes
        const uint8_t *data = static_cast<const uint8_t *>
                              (output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          if (data[idx] != 0) {
            has_non_zero = true;
            break;
          }
        }
      }
      return has_non_zero;
    }
    return true;
  }

  // -----------------------------------------------------------------------------
  // validate_boundary_output
  //
  // Validates boundary-specific output conditions (NaN/Inf checks, value reasonableness).
  // Used in boundary tests to ensure output tensor is numerically reasonable and free of NaN/Inf.
  //
  // Parameters:
  //   output - Output tensor
  //   params - Matmul test parameters
  // Returns:
  //   true if output tensor passes boundary checks, false otherwise
  // -----------------------------------------------------------------------------
  bool validate_boundary_output(const tensor_t &output,
                                const MatmulParamsAI &params) {
    // Check basic tensor properties first (dimensions, dtype, nonzero)
    // If these fail, output is not valid
    if (!validate_output_tensor(output, params)) {
      return false;
    }

    // --- BOUNDARY-SPECIFIC VALIDATION LOGIC ---
    // If output is empty, nothing to check
    if (output.get_nelem() == 0) {
      return true;
    }

    auto dtype = output.get_data_type();
    size_t num_elements = output.get_nelem();
    auto sample_indices = AITestUtils::get_sample_indices(num_elements, 1000);

    bool has_nan = false;
    bool has_inf = false;
    bool has_reasonable_values = false;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    if (dtype == data_type_t::f32) {
      const float *data = static_cast<const float *>(output.get_raw_handle_const());
      for (size_t idx : sample_indices) {
        float val = data[idx];
        if (std::isnan(val)) {
          has_nan = true;
        }
        if (std::isinf(val)) {
          has_inf = true;
        }
        if (std::isfinite(val)) {
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
          if (std::abs(val) > 1e-10f && std::abs(val) < 1e10f) {
            has_reasonable_values = true;
          }
        }
      }
    }
    else if (dtype == data_type_t::bf16) {
      const bfloat16_t *data = static_cast<const bfloat16_t *>
                               (output.get_raw_handle_const());
      for (size_t idx : sample_indices) {
        float val = static_cast<float>(data[idx]);
        // NaN/Inf checks for bf16
        if (std::isnan(val)) {
          has_nan = true;
        }
        if (std::isinf(val)) {
          has_inf = true;
        }
        if (std::isfinite(val)) {
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
          if (std::abs(val) > 1e-7f && std::abs(val) < 1e5f) {
            has_reasonable_values = true;
          }
        }
      }
    }
    else if (dtype == data_type_t::s8 || dtype == data_type_t::s4) {
      const int8_t *data = static_cast<const int8_t *>(output.get_raw_handle_const());
      for (size_t idx : sample_indices) {
        int8_t val = data[idx];
        // For int8, NaN/Inf not applicable, just check range
        min_val = std::min<float>(min_val, val);
        max_val = std::max<float>(max_val, val);
        if (val != 0) {
          has_reasonable_values = true;
        }
      }
    }
    else {
      // For other types, fallback: check raw bytes
      const uint8_t *data = static_cast<const uint8_t *>
                            (output.get_raw_handle_const());
      for (size_t idx : sample_indices) {
        uint8_t val = data[idx];
        if (val != 0) {
          has_reasonable_values = true;
        }
      }
    }
    // Fail if any NaN or Inf is found (should never happen in correct matmul)
    if (has_nan) {
      std::cout << "[AI_BOUNDARY_ERROR] Output contains NaN values" << std::endl;
      return false;
    }
    if (has_inf) {
      std::cout << "[AI_BOUNDARY_ERROR] Output contains Inf values" << std::endl;
      return false;
    }
    // For very small matrices (e.g., 1x1, vectors), require at least one reasonable value
    if (params.m * params.n <= 16 && !has_reasonable_values) {
      std::cout <<
                "[AI_BOUNDARY_ERROR] Small boundary case produced unreasonable values" <<
                std::endl;
      return false;
    }
    // If all checks pass, output is considered valid for boundary conditions
    return true;
  }

  // -----------------------------------------------------------------------------
  // validate_numerical_stability
  //
  // Validates numerical stability for boundary condition tests (relative error, RMS, range).
  // Used in boundary tests to check for overflow, underflow, and gradient explosion in output.
  //
  // Parameters:
  //   output - Output tensor
  //   input - Input tensor
  //   weights - Weights tensor
  //   params - Matmul test parameters
  // Returns:
  //   true if output is numerically stable, false otherwise
  // -----------------------------------------------------------------------------
  bool validate_numerical_stability(const CreatedTensors &tensors,
                                    const MatmulParamsAI &params) {
    // --- NUMERICAL STABILITY CHECKS FOR BOUNDARY CONDITIONS ---
    // If output is empty, nothing to check
    if (tensors.output.get_nelem() == 0) {
      return true;
    }

    auto dtype = tensors.output.get_data_type();
    auto input_dtype = tensors.input.get_data_type();
    auto weight_dtype = tensors.weights.get_data_type();
    auto bias_dtype = tensors.bias.get_data_type();
    auto sample_indices = AITestUtils::get_sample_indices(
                            tensors.output.get_nelem(), 100);

    // Helper lambda to extract value as float from any supported type
    auto get_val = [](const void *ptr, data_type_t dtype) -> float {
      switch (dtype) {
      case data_type_t::f32:
        return static_cast<const float *>(ptr)[0];
      case data_type_t::bf16:
        return static_cast<float>(static_cast<const bfloat16_t *>(ptr)[0]);
      case data_type_t::s8:
      case data_type_t::s4:
        return static_cast<float>(static_cast<const int8_t *>(ptr)[0]);
      default:
        return static_cast<float>(static_cast<const uint8_t *>(ptr)[0]);
      }
    };

    // 1. For 1x1x1 matmul with no post-ops, check that the output matches the expected value (within 1% relative error)
    if (params.m == 1 && params.n == 1 && params.k == 1 &&
        params.post_op_config.post_ops.empty()) {
      float input_val = get_val(tensors.input.get_raw_handle_const(), input_dtype);
      float weight_val = get_val(tensors.weights.get_raw_handle_const(),
                                 weight_dtype);
      float bias_val = get_val(tensors.bias.get_raw_handle_const(), bias_dtype);
      float expected_val = input_val * weight_val + bias_val;
      float actual_val = get_val(tensors.output.get_raw_handle_const(), dtype);
      float expected_magnitude = std::abs(expected_val);
      float actual_magnitude = std::abs(actual_val);
      float relative_error = std::abs(actual_magnitude - expected_magnitude) /
                             (expected_magnitude + 1e-10f);
      if (relative_error > 0.01f) {
        return false;
      }
    }

    // 2. For vector outputs (m==1 or n==1), check that the output RMS magnitude is not too large or too small
    if (params.m == 1 || params.n == 1) {
      float output_magnitude = 0.0f;
      if (dtype == data_type_t::f32) {
        const float *output_data = static_cast<const float *>
                                   (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          output_magnitude += output_data[idx] * output_data[idx];
        }
      }
      else if (dtype == data_type_t::bf16) {
        const bfloat16_t *output_data = static_cast<const bfloat16_t *>
                                        (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          output_magnitude += val * val;
        }
      }
      else if (dtype == data_type_t::s8 || dtype == data_type_t::s4) {
        const int8_t *output_data = static_cast<const int8_t *>
                                    (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          output_magnitude += val * val;
        }
      }
      else {
        const uint8_t *output_data = static_cast<const uint8_t *>
                                     (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          output_magnitude += val * val;
        }
      }
      output_magnitude = std::sqrt(output_magnitude / sample_indices.size());
      if (output_magnitude > 1000.0f) {
        std::cout <<
                  "[AI_BOUNDARY_ERROR] Vector operation produced excessive magnitude: "
                  << output_magnitude << std::endl;
        return false;
      }
      if (output_magnitude < 1e-10f && params.k > 1) {
        std::cout <<
                  "[AI_BOUNDARY_ERROR] Vector operation produced suspiciously small magnitude: "
                  << output_magnitude << std::endl;
        return false;
      }
    }

    // 3. For large outputs, check that the range (max-min) is not excessive (gradient explosion)
    if (params.m * params.n > 100) {
      float max_output = -std::numeric_limits<float>::max();
      float min_output = std::numeric_limits<float>::max();
      if (dtype == data_type_t::f32) {
        const float *output_data = static_cast<const float *>
                                   (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          max_output = std::max(max_output, output_data[idx]);
          min_output = std::min(min_output, output_data[idx]);
        }
      }
      else if (dtype == data_type_t::bf16) {
        const bfloat16_t *output_data = static_cast<const bfloat16_t *>
                                        (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          max_output = std::max(max_output, val);
          min_output = std::min(min_output, val);
        }
      }
      else if (dtype == data_type_t::s8 || dtype == data_type_t::s4) {
        const int8_t *output_data = static_cast<const int8_t *>
                                    (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          max_output = std::max(max_output, val);
          min_output = std::min(min_output, val);
        }
      }
      else {
        const uint8_t *output_data = static_cast<const uint8_t *>
                                     (tensors.output.get_raw_handle_const());
        for (size_t idx : sample_indices) {
          float val = static_cast<float>(output_data[idx]);
          max_output = std::max(max_output, val);
          min_output = std::min(min_output, val);
        }
      }
      float output_range = max_output - min_output;
      if (output_range > 1000.0f) {
        std::cout << "[AI_BOUNDARY_ERROR] Large output range detected: " << output_range
                  << std::endl;
        return false;
      }
    }

    std::cout << "[AI_BOUNDARY_OK] Numerical stability validated" << std::endl;
    return true;
  }
};

/**
 * @brief Main test method for ZenDNNL matmul AI tests.
 *        Routes to the appropriate test type based on category (accuracy, boundary, edge case,
 *        invalid, reference kernel). Used by all test instantiations to run the correct test flow.
 */
TEST_P(TestMatmulAI, ComprehensiveMatmulTest) {
  MatmulParamsAI params = GetParam();

  // Validate data type combination is supported
  if (!AITestUtils::is_valid_data_type_combination(params.data_types)) {
    GTEST_SKIP() << "Data type combination not yet supported: "
                 << static_cast<int>(params.data_types);
    return;
  }

  // Route to appropriate test based on category
  switch (params.category) {
  case TestCategory::ACCURACY:
    run_accuracy_test(params);
    break;
  case TestCategory::BOUNDARY:
    run_boundary_test(params);
    break;
  case TestCategory::EDGE_CASE:
    run_edge_case_test(params);
    break;
  case TestCategory::INVALID:
    run_invalid_test(params);
    break;
  case TestCategory::REFERENCE_KERNEL:
    run_reference_kernel_test(params);
    break;
  default:
    FAIL() << "Unknown test category: " << static_cast<int>(params.category);
    break;
  }
}

// Single test instantiation based on global test mode
// The test mode (PRE_SUB, POST_SUB, NIGHTLY) is determined by the global variable ai_gtest_mode
// which can be set externally before test instantiation
INSTANTIATE_TEST_SUITE_P(
  AITests,
  TestMatmulAI,
  ::testing::ValuesIn(
    get_test_suite_for_mode<MatmulParamsAI, ParameterGenerator>()),
[](const ::testing::TestParamInfo<MatmulParamsAI> &info) {
  return info.param.test_name;
}
);
