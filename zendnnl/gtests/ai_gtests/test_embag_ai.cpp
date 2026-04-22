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
#include "gtest_utils_embag_ai.hpp"
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

using namespace ai_gtests;
using namespace zendnnl::lowoha::embag;
#include <iostream>
#include <memory>
#include <cstdlib>

/** @brief AI Test class for comprehensive ZenDNNL embedding bag testing */
class TestEmbagAI : public ::testing::TestWithParam<EmbagParamsAI> {
  // -----------------------------------------------------------------------------
  // CreatedTensors
  //
  // Helper struct for storing all tensors created for an embedding bag test.
  // Used to pass table, indices, offsets, output, and reference output tensors
  // between test flows. All test flows use this struct for consistency.
  // -----------------------------------------------------------------------------
  struct CreatedTensors {
    tensor_t table;
    tensor_t indices;
    tensor_t offsets;
    tensor_t output;
    tensor_t reference_output; // optional, may be empty
  };

  // -----------------------------------------------------------------------------
  // create_test_tensors
  //
  // Centralized tensor creation helper for all embedding bag test flows.
  // Creates table, indices, offsets, output, and reference output tensors
  // with standardized category-based naming. Handles both embedding bag mode
  // (with offsets) and embedding lookup mode (without offsets).
  //
  // Parameters:
  //   params - Embedding bag test parameters
  //   table_dtype - Data type for embedding table
  //   output_dtype - Data type for output tensor
  //   use_boundary - If true, create boundary tensors; else, uniform tensors
  //   create_reference_output - If true, create reference output tensor
  //   name_suffix - Suffix for tensor names (category-based)
  // Returns:
  //   Struct containing all created tensors
  // -----------------------------------------------------------------------------
  CreatedTensors create_test_tensors(const EmbagParamsAI &params,
                                     data_type_t table_dtype,
                                     data_type_t output_dtype,
                                     bool use_boundary = false,
                                     bool create_reference_output = false,
                                     const std::string &name_suffix = "") {
    CreatedTensors tensors;

    std::string table_name = "table" + name_suffix;
    std::string indices_name = "indices" + name_suffix;
    std::string offsets_name = "offsets" + name_suffix;
    std::string output_name = "output_zero" + name_suffix;
    std::string ref_output_name = "ref_output_zero" + name_suffix;

    // Get dimensions
    auto table_dims = EmbagTestUtils::get_table_dims(params.num_embeddings,
                      params.embedding_dim);
    auto indices_dims = EmbagTestUtils::get_indices_dims(params.num_indices);
    auto output_dims = params.use_offsets ?
                       EmbagTestUtils::get_output_dims_bag(params.num_bags, params.embedding_dim) :
                       EmbagTestUtils::get_output_dims_lookup(params.num_indices,
                           params.embedding_dim);

    // Create table tensor
    if (table_dtype == data_type_t::u4 || table_dtype == data_type_t::s8 ||
        table_dtype == data_type_t::s4) {
      // For quantized tables, use special quantized creation method
      tensors.table = AITensorFactory::create_quantized_embedding_tensor(
                        table_dims, table_dtype, table_name, params.fp16_scale_bias);
    }
    else {
      auto tensor_creator = use_boundary ? &AITensorFactory::create_boundary_tensor
                            : &AITensorFactory::create_uniform_tensor;
      tensors.table = tensor_creator(table_dims, table_dtype, table_name);
    }

    // Create indices tensor
    std::vector<int64_t> indices_data;
    if (use_boundary) {
      indices_data = EmbagTestUtils::generate_boundary_indices(params.num_indices,
                     params.num_embeddings);
    }
    else {
      indices_data = EmbagTestUtils::generate_random_indices(params.num_indices,
                     params.num_embeddings,
                     params.use_padding_idx ? params.padding_idx : -1);
    }

    data_type_t indices_dtype = EmbagTestUtils::get_indices_dtype(
                                  params.indices_type);
    std::vector<tensor_t::index_type> indices_size_vec(indices_dims.begin(),
        indices_dims.end());
    tensors.indices = tensor_t()
                      .set_name(indices_name)
                      .set_size(indices_size_vec)
                      .set_data_type(indices_dtype)
                      .set_storage()
                      .create();

    if (!tensors.indices.check()) {
      throw std::runtime_error("Failed to create indices tensor");
    }
    void *indices_ptr = tensors.indices.get_raw_handle_unsafe();
    if (!indices_ptr) {
      throw std::runtime_error("Null pointer for indices tensor");
    }

    // Copy indices data based on indices data type
    if (indices_dtype == data_type_t::s32) {
      // Convert int64_t to int32_t
      std::vector<int32_t> indices_data_s32(indices_data.size());
      for (size_t i = 0; i < indices_data.size(); ++i) {
        indices_data_s32[i] = static_cast<int32_t>(indices_data[i]);
      }
      std::memcpy(indices_ptr, indices_data_s32.data(),
                  indices_data_s32.size() * sizeof(int32_t));
    }
    else {
      std::memcpy(indices_ptr, indices_data.data(),
                  indices_data.size() * sizeof(int64_t));
    }

    // Create offsets tensor if needed
    if (params.use_offsets) {
      auto offsets_dims = EmbagTestUtils::get_offsets_dims(params.num_bags,
                          params.include_last_offset);
      std::vector<int64_t> offsets_data = EmbagTestUtils::generate_offsets(
                                            params.num_bags, params.num_indices, params.include_last_offset);

      std::vector<tensor_t::index_type> offsets_size_vec(offsets_dims.begin(),
          offsets_dims.end());
      tensors.offsets = tensor_t()
                        .set_name(offsets_name)
                        .set_size(offsets_size_vec)
                        .set_data_type(indices_dtype)
                        .set_storage()
                        .create();

      if (!tensors.offsets.check()) {
        throw std::runtime_error("Failed to create offsets tensor");
      }
      void *offsets_ptr = tensors.offsets.get_raw_handle_unsafe();
      if (!offsets_ptr) {
        throw std::runtime_error("Null pointer for offsets tensor");
      }

      // Copy offsets data based on indices data type
      if (indices_dtype == data_type_t::s32) {
        // Convert int64_t to int32_t
        std::vector<int32_t> offsets_data_s32(offsets_data.size());
        for (size_t i = 0; i < offsets_data.size(); ++i) {
          offsets_data_s32[i] = static_cast<int32_t>(offsets_data[i]);
        }
        std::memcpy(offsets_ptr, offsets_data_s32.data(),
                    offsets_data_s32.size() * sizeof(int32_t));
      }
      else {
        std::memcpy(offsets_ptr, offsets_data.data(),
                    offsets_data.size() * sizeof(int64_t));
      }
    }

    // Create output tensors
    tensors.output = AITensorFactory::create_zero_tensor(output_dims, output_dtype,
                     output_name);
    if (create_reference_output) {
      tensors.reference_output = AITensorFactory::create_zero_tensor(output_dims,
                                 output_dtype, ref_output_name);
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
    // Add specific setup code here if needed
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
  //   params - Embedding bag test parameters
  // -----------------------------------------------------------------------------
  void run_reference_kernel_test(const EmbagParamsAI &params) {
    ASSERT_TRUE(EmbagTestUtils::validate_embag_dimensions(params.num_embeddings,
                params.embedding_dim, params.num_indices, params.num_bags))
        << "Invalid dimensions for reference kernel test";

    auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
    auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                       false, false, "_reference");

    // Run reference kernel only
    status_t ref_status = EmbagTestUtils::run_reference_embag(
                            tensors.table, tensors.indices, tensors.offsets,
                            tensors.output, params);

    if (EmbagTestUtils::is_embag_reference_supported(table_dtype, output_dtype,
        params.algo)) {
      EXPECT_EQ(ref_status, status_t::success)
          << "Reference kernel must succeed for supported dtype/algo combinations";
    }
    else {
      EXPECT_NE(ref_status, status_t::success)
          << "Reference kernel should fail for non-supported combinations";
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
    return EmbagTestUtils::get_accuracy_tolerance(output_dtype);
  }

  float get_relative_tolerance(data_type_t output_dtype) const {
    return EmbagTestUtils::get_relative_tolerance(output_dtype);
  }

  // -----------------------------------------------------------------------------
  // run_accuracy_test
  //
  // Runs accuracy test by comparing ZenDNNL kernel output against reference implementation.
  // Handles both reference-supported and non-reference-supported cases.
  // Used for validating correctness of ZenDNNL kernel output.
  //
  // Parameters:
  //   params - Embedding bag test parameters
  // -----------------------------------------------------------------------------
  void run_accuracy_test(const EmbagParamsAI &params) {
    ASSERT_TRUE(EmbagTestUtils::validate_embag_dimensions(params.num_embeddings,
                params.embedding_dim, params.num_indices, params.num_bags))
        << "Invalid dimensions for accuracy test";

    auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
    auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                       false, true, "_accuracy");

    AITestUtils::debug_print("[AI_EMBAG_DEBUG] About to run ZenDNNL kernel...");
    bool kernel_supported = EmbagTestUtils::is_embag_kernel_supported(table_dtype,
                            output_dtype, params.algo);
    status_t test_status = run_embag_test(tensors.table, tensors.indices,
                                          tensors.offsets, tensors.output, params);
    AITestUtils::debug_print("[AI_EMBAG_DEBUG] ZenDNNL kernel finished.");

    if (kernel_supported) {
      EXPECT_EQ(test_status, status_t::success)
          << "ZenDNNL kernel must succeed for supported data types";
    }
    else {
      EXPECT_NE(test_status, status_t::success)
          << "ZenDNNL kernel should fail for unsupported data types";
    }

    // **ACCURACY TEST LOGIC BASED ON REFERENCE IMPLEMENTATION SUPPORT**
    if (EmbagTestUtils::is_embag_reference_supported(table_dtype, output_dtype,
        params.algo)) {
      // **REFERENCE SUPPORTED: ZenDNNL must succeed AND match reference implementation**
      if (test_status == status_t::success) {
        // Run reference implementation for accuracy comparison
        AITestUtils::debug_print("[AI_EMBAG_DEBUG] About to run reference implementation...");
        status_t ref_status = EmbagTestUtils::run_reference_embag(
                                tensors.table, tensors.indices, tensors.offsets,
                                tensors.reference_output, params);
        AITestUtils::debug_print("[AI_EMBAG_DEBUG] Reference implementation finished.");

        EXPECT_EQ(ref_status, status_t::success)
            << "Reference implementation must succeed for supported data types";

        if (ref_status == status_t::success) {
          // Compare ZenDNNL output with reference implementation
          float abs_tolerance = get_accuracy_tolerance(output_dtype);
          float rel_tolerance = get_relative_tolerance(output_dtype);
          bool comparison_result = EmbagTestUtils::compare_embag_tensors(
                                     tensors.output, tensors.reference_output,
                                     abs_tolerance, rel_tolerance);
          EXPECT_TRUE(comparison_result)
              << "ZenDNNL output must match reference within tolerance, abs: "
              << abs_tolerance << ", rel: " << rel_tolerance;
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
  // Runs boundary condition test for embedding bag (e.g., extreme values, edge dimensions).
  // Validates output for numerical stability and correctness. Used to test kernel
  // behavior under boundary conditions and extreme values.
  //
  // Parameters:
  //   params - Embedding bag test parameters
  // -----------------------------------------------------------------------------
  void run_boundary_test(const EmbagParamsAI &params) {
    auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
    auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                       true, false, "_boundary");

    status_t test_status = run_embag_test(tensors.table, tensors.indices,
                                          tensors.offsets, tensors.output, params);

    if (params.expect_success) {
      EXPECT_EQ(test_status, status_t::success)
          << "Boundary test failed when success was expected for " << params.test_name;

      // **BOUNDARY-SPECIFIC VALIDATION**
      EXPECT_TRUE(validate_boundary_output(tensors.output, params))
          << "Boundary output validation failed for " << params.test_name;
    }
    else {
      EXPECT_NE(test_status, status_t::success)
          << "Boundary test succeeded when failure was expected for " << params.test_name;
    }

    std::cout << "[AI_EMBAG_BOUNDARY] " << params.test_name <<
              " completed with status: "
              << static_cast<int>(test_status) << std::endl;
  }

  /**
   * @brief Runs edge case test for embedding bag (e.g., unusual but valid shapes).
   *        Validates kernel success/failure as expected. Used to test kernel robustness
   *        for edge-case shapes and parameters.
   *
   * @param params Embedding bag test parameters
   */
  void run_edge_case_test(const EmbagParamsAI &params) {
    auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
    auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                       false, false, "_edge");

    status_t test_status = run_embag_test(tensors.table, tensors.indices,
                                          tensors.offsets, tensors.output, params);

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
   * @brief Runs invalid case test for embedding bag (e.g., invalid shapes, missing tensors).
   *        Expects kernel to fail for all invalid scenarios. Used to verify kernel error
   *        handling and robustness against invalid inputs.
   *
   * @param params Embedding bag test parameters
   */
  void run_invalid_test(const EmbagParamsAI &params) {
    // For invalid tests, we expect failures
    EXPECT_FALSE(params.expect_success)
        << "Invalid test should have expect_success = false";

    // **RUN ACTUAL EMBEDDING BAG KERNEL WITH INVALID DIMENSIONS**
    status_t test_status = run_invalid_embag_test(params);

    // Invalid tests should fail
    EXPECT_NE(test_status, status_t::success)
        << "Invalid test should fail but succeeded: " << params.test_name;
  }

  // -----------------------------------------------------------------------------
  // run_group_embag_test
  //
  // Runs group embedding bag test using group_embedding_bag_direct API.
  // Creates multiple sets of tensors and runs batched embedding bag operation.
  // Used for testing the grouped/batched embedding bag direct API.
  //
  // Parameters:
  //   params - Embedding bag test parameters (with is_group_embag=true)
  // -----------------------------------------------------------------------------
  void run_group_embag_test(const EmbagParamsAI &params) {
    ASSERT_TRUE(params.is_group_embag)
        << "run_group_embag_test called with non-group params";
    ASSERT_GT(params.group_size, 0)
        << "Group size must be greater than 0";

    auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
    auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

    // Vectors to hold tensor groups and raw pointers
    std::vector<CreatedTensors> tensor_groups;
    std::vector<const void *> tables;
    std::vector<const void *> indices;
    std::vector<const void *> offsets;
    std::vector<const float *> weights;
    std::vector<void *> dsts;
    std::vector<embag_params_t> embag_params_vec;

    // Create tensors for each table in the group
    for (uint64_t i = 0; i < params.group_size; ++i) {
      std::string suffix = "_group" + std::to_string(i);
      auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                         false, false, suffix);
      tensor_groups.push_back(tensors);

      // Build raw pointers
      tables.push_back(tensors.table.get_raw_handle_const());
      indices.push_back(tensors.indices.get_raw_handle_const());
      if (params.use_offsets) {
        offsets.push_back(tensors.offsets.get_raw_handle_const());
      }
      else {
        offsets.push_back(nullptr);
      }
      weights.push_back(nullptr);  // AI tests don't use per-sample weights
      dsts.push_back(tensors.output.get_raw_handle_unsafe());

      // Build embag_params_t for this table
      data_type_t indices_dtype = EmbagTestUtils::get_indices_dtype(
                                    params.indices_type);
      embag_params_t embag_params_obj;
      embag_params_obj.dtypes.table = table_dtype;
      embag_params_obj.dtypes.output = output_dtype;
      embag_params_obj.dtypes.indices = indices_dtype;
      if (params.use_offsets) {
        embag_params_obj.dtypes.offsets = indices_dtype;
      }

      embag_params_obj.algo = params.algo;
      embag_params_obj.num_embeddings = params.num_embeddings;
      embag_params_obj.embedding_dim = params.embedding_dim;
      embag_params_obj.num_indices = params.num_indices;

      if (params.use_offsets) {
        embag_params_obj.num_bags = params.num_bags;
        embag_params_obj.include_last_offset = params.include_last_offset;
      }

      embag_params_obj.is_weights = false;
      embag_params_obj.padding_idx = params.use_padding_idx ? params.padding_idx : -1;
      embag_params_obj.num_threads = 0;
      embag_params_obj.fp16_scale_bias = params.fp16_scale_bias;
      embag_params_obj.dst_stride = tensors.output.get_stride()[0];

      embag_params_vec.push_back(embag_params_obj);
    }

    // Call group_embedding_bag_direct
    AITestUtils::debug_print("[AI_EMBAG_DEBUG] Running group_embedding_bag_direct with "
                             + std::to_string(params.group_size) + " tables...");

    status_t status = group_embedding_bag_direct(
                        tables, indices, offsets, weights, dsts, embag_params_vec);

    AITestUtils::debug_print("[AI_EMBAG_DEBUG] group_embedding_bag_direct finished.");

    if (params.expect_success) {
      EXPECT_EQ(status, status_t::success)
          << "Group embedding bag test failed when success was expected for "
          << params.test_name;
    }
    else {
      EXPECT_NE(status, status_t::success)
          << "Group embedding bag test succeeded when failure was expected for "
          << params.test_name;
    }

    std::cout << "[AI_EMBAG_GROUP] " << params.test_name
              << " with " << params.group_size << " tables completed with status: "
              << static_cast<int>(status) << std::endl;
  }

 private:
  // -----------------------------------------------------------------------------
  // run_invalid_embag_test
  //
  // Simulates all invalid embedding bag scenarios and verifies kernel failure.
  // Handles specific cases (invalid dimensions, out-of-range indices, etc.).
  // Used internally by run_invalid_test to cover all negative code paths.
  //
  // Parameters:
  //   params - Embedding bag test parameters
  // Returns:
  //   Failure status if kernel behaves correctly
  // -----------------------------------------------------------------------------
  status_t run_invalid_embag_test(const EmbagParamsAI &params) {
    try {
      auto table_dtype = EmbagTestUtils::get_table_dtype(params.data_types);
      auto output_dtype = EmbagTestUtils::get_output_dtype(params.data_types);

      // Try to create tensors with invalid dimensions
      auto tensors = create_test_tensors(params, table_dtype, output_dtype,
                                         false, false, "_invalid");

      // Try to run the operator
      status_t status = run_embag_test(tensors.table, tensors.indices,
                                       tensors.offsets, tensors.output, params);
      return status;
    }
    catch (...) {
      return status_t::failure;
    }
  }

  // -----------------------------------------------------------------------------
  // run_embag_test
  //
  // Core embedding bag test execution for ZenDNNL kernel.
  // Runs ZenDNNL embedding bag kernel with provided tensors and parameters.
  // Used by all test flows except reference-only tests.
  //
  // Parameters:
  //   table - Embedding table tensor
  //   indices - Indices tensor
  //   offsets - Offsets tensor (may be empty for embedding lookup)
  //   output - Output tensor
  //   params - Embedding bag test parameters
  // Returns:
  //   Success or failure status
  // -----------------------------------------------------------------------------
  status_t run_embag_test(tensor_t &table, tensor_t &indices,
                          tensor_t &offsets, tensor_t &output,
                          const EmbagParamsAI &params) {
    try {
      // Check if LOWOHA mode is enabled
      if (is_lowoha_mode_enabled()) {
        // LOWOHA path - use embedding_bag_direct or embedding_direct API
        AITestUtils::debug_print("[AI_EMBAG_DEBUG] Running LOWOHA embedding_bag_direct...");

        // Validate tensors
        if (!table.check() || !indices.check() || !output.check()) {
          std::cout << "[AI_EMBAG_TEST] LOWOHA: Invalid tensor state for " <<
                    params.test_name << std::endl;
          return status_t::failure;
        }

        if (params.use_offsets && !offsets.check()) {
          std::cout << "[AI_EMBAG_TEST] LOWOHA: Invalid offsets tensor for " <<
                    params.test_name << std::endl;
          return status_t::failure;
        }

        // Get data pointers
        void *table_data = table.get_raw_handle_unsafe();
        void *indices_data = indices.get_raw_handle_unsafe();
        void *offsets_data = params.use_offsets ? offsets.get_raw_handle_unsafe() :
                             nullptr;
        void *output_data = output.get_raw_handle_unsafe();

        if (!table_data || !indices_data || !output_data) {
          std::cout << "[AI_EMBAG_TEST] LOWOHA: Null data pointer for " <<
                    params.test_name << std::endl;
          return status_t::failure;
        }

        // Build embag_params_t structure
        embag_params_t embag_params_obj;
        embag_params_obj.dtypes.table = table.get_data_type();
        embag_params_obj.dtypes.output = output.get_data_type();
        embag_params_obj.dtypes.indices = indices.get_data_type();
        if (params.use_offsets) {
          embag_params_obj.dtypes.offsets = offsets.get_data_type();
        }

        embag_params_obj.algo = params.algo;
        embag_params_obj.num_embeddings = params.num_embeddings;
        embag_params_obj.embedding_dim = params.embedding_dim;
        embag_params_obj.num_indices = params.num_indices;

        if (params.use_offsets) {
          embag_params_obj.num_bags = params.num_bags;
          embag_params_obj.include_last_offset = params.include_last_offset;
        }

        embag_params_obj.is_weights = false;  // AI tests don't use per-sample weights
        embag_params_obj.padding_idx = params.use_padding_idx ? params.padding_idx : -1;
        embag_params_obj.num_threads = 0;  // Use default
        embag_params_obj.fp16_scale_bias = params.fp16_scale_bias;
        embag_params_obj.dst_stride = output.get_stride()[0];

        // Call appropriate LOWOHA API
        status_t status;
        if (params.use_offsets) {
          // Embedding bag with offsets
          status = embedding_bag_direct(
                     table_data,
                     indices_data,
                     offsets_data,
                     nullptr,  // weights_data (not used in AI tests)
                     output_data,
                     embag_params_obj);
        }
        else {
          // Embedding lookup (no offsets)
          status = embedding_direct(
                     table_data,
                     indices_data,
                     nullptr,  // weights_data (not used in AI tests)
                     output_data,
                     embag_params_obj);
        }

        if (status != status_t::success) {
          std::cout << "[AI_EMBAG_TEST] LOWOHA embedding API failed for " <<
                    params.test_name << std::endl;
        }
        return status;
      }
      else {
        // Primitive operator path (default)
        AITestUtils::debug_print("[AI_EMBAG_DEBUG] Running primitive embag operator...");

        auto embag_context = embag_context_t()
                             .set_param("table", table)
                             .set_algo(params.algo);

        if (params.use_padding_idx) {
          embag_context = embag_context.set_padding_index(params.padding_idx);
        }
        if (params.fp16_scale_bias) {
          embag_context = embag_context.set_fp16_scale_bias(true);
        }
        if (params.include_last_offset) {
          embag_context = embag_context.set_include_last_offset(true);
        }

        embag_context = embag_context.create();
        if (!embag_context.check()) {
          std::cout << "[AI_EMBAG_TEST] Context creation failed for " << params.test_name
                    << std::endl;
          return status_t::failure;
        }

        auto embag_operator = embag_operator_t()
                              .set_name(AITestUtils::generate_unique_name("embag_ai_op"))
                              .set_context(embag_context)
                              .create();

        if (embag_operator.is_bad_object()) {
          std::cout << "[AI_EMBAG_TEST] Operator creation failed for " << params.test_name
                    << std::endl;
          return status_t::failure;
        }

        embag_operator = embag_operator.set_input("indices", indices);
        if (params.use_offsets) {
          embag_operator = embag_operator.set_input("offsets", offsets);
        }
        embag_operator = embag_operator.set_output("output", output);

        auto status = embag_operator.execute();
        return status;
      }
    }
    catch (const std::exception &e) {
      std::cout << "[AI_EMBAG_TEST] Exception in " << params.test_name << ": "
                << e.what() << std::endl;
      return status_t::failure;
    }
    catch (...) {
      std::cout << "[AI_EMBAG_TEST] Unknown exception in " << params.test_name
                << std::endl;
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
  //   params - Embedding bag test parameters
  // Returns:
  //   true if output tensor is valid, false otherwise
  // -----------------------------------------------------------------------------
  bool validate_output_tensor(const tensor_t &output,
                              const EmbagParamsAI &params) {
    // Check dimensions
    auto expected_dims = params.use_offsets ?
                         EmbagTestUtils::get_output_dims_bag(params.num_bags, params.embedding_dim) :
                         EmbagTestUtils::get_output_dims_lookup(params.num_indices,
                             params.embedding_dim);

    if (output.get_size() != expected_dims) {
      return false;
    }

    // Check data type
    auto expected_dtype = EmbagTestUtils::get_output_dtype(params.data_types);
    if (output.get_data_type() != expected_dtype) {
      std::cout << "[AI_EMBAG_BOUNDARY_ERROR] Output tensor data type mismatch" <<
                std::endl;
      return false;
    }

    // Check that output is not all zeros (basic sanity check)
    // Skip this check for very small embedding dimensions in boundary tests
    // as boundary values might cancel out or produce very small results
    if (output.get_nelem() > 0 && params.embedding_dim > 2) {
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

      if (!has_non_zero) {
        std::cout << "[AI_EMBAG_BOUNDARY_ERROR] Output is all zeros" << std::endl;
        return false;
      }
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
  //   params - Embedding bag test parameters
  // Returns:
  //   true if output tensor passes boundary checks, false otherwise
  // -----------------------------------------------------------------------------
  bool validate_boundary_output(const tensor_t &output,
                                const EmbagParamsAI &params) {
    if (!validate_output_tensor(output, params)) {
      return false;
    }

    if (output.get_nelem() == 0) {
      return true;
    }

    auto dtype = output.get_data_type();
    size_t num_elements = output.get_nelem();
    auto sample_indices = AITestUtils::get_sample_indices(num_elements, 1000);

    bool has_nan = false;
    bool has_inf = false;
    bool has_reasonable_values = false;

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
        if (std::isfinite(val) && std::abs(val) > 1e-10f && std::abs(val) < 1e10f) {
          has_reasonable_values = true;
        }
      }
    }
    else if (dtype == data_type_t::bf16) {
      const bfloat16_t *data = static_cast<const bfloat16_t *>
                               (output.get_raw_handle_const());
      for (size_t idx : sample_indices) {
        float val = static_cast<float>(data[idx]);
        if (std::isnan(val)) {
          has_nan = true;
        }
        if (std::isinf(val)) {
          has_inf = true;
        }
        if (std::isfinite(val) && std::abs(val) > 1e-7f && std::abs(val) < 1e5f) {
          has_reasonable_values = true;
        }
      }
    }

    if (has_nan) {
      std::cout << "[AI_EMBAG_BOUNDARY_ERROR] Output contains NaN values" <<
                std::endl;
      return false;
    }
    if (has_inf) {
      std::cout << "[AI_EMBAG_BOUNDARY_ERROR] Output contains Inf values" <<
                std::endl;
      return false;
    }

    // For small boundary cases, ensure we have reasonable values
    // Skip this check for very small embedding dimensions (≤2) as boundary values
    // in such cases can legitimately produce edge-case results
    uint64_t total_elements = params.use_offsets ?
                              (params.num_bags * params.embedding_dim) :
                              (params.num_indices * params.embedding_dim);
    if (total_elements <= 1000 && params.embedding_dim > 2 &&
        !has_reasonable_values) {
      std::cout <<
                "[AI_EMBAG_BOUNDARY_ERROR] Small boundary case produced unreasonable values"
                << std::endl;
      return false;
    }

    return true;
  }
};

/**
 * @brief Main test method for ZenDNNL embedding bag AI tests.
 *        Routes to the appropriate test type based on category (accuracy, boundary, edge case,
 *        invalid, reference kernel, group embedding bag). Used by all test instantiations to run the correct test flow.
 */
TEST_P(TestEmbagAI, ComprehensiveEmbagTest) {
  EmbagParamsAI params = GetParam();

  // Validate data type combination is supported
  if (!EmbagTestUtils::is_valid_embag_data_type_combination(params.data_types)) {
    GTEST_SKIP() << "Data type combination not yet supported: "
                 << static_cast<int>(params.data_types);
    return;
  }

  // Check for group embedding bag test first (special handling)
  if (params.is_group_embag) {
    run_group_embag_test(params);
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
  TestEmbagAI,
  ::testing::ValuesIn(
    get_test_suite_for_mode<EmbagParamsAI, EmbagParameterGenerator>()),
[](const ::testing::TestParamInfo<EmbagParamsAI> &info) {
  return info.param.test_name;
}
);
