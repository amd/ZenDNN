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

#include "gtest_utils_embag_ai.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <set>
#include <iostream>

using namespace zendnnl::memory;
using namespace zendnnl::common;
using namespace zendnnl::ops;
using namespace zendnnl::error_handling;

namespace ai_gtests {

// Random number generation
static std::mt19937 embag_rng(
  std::chrono::steady_clock::now().time_since_epoch().count());

// Helper function to generate random dimensions within a range
static uint64_t generate_random_embag_dim(uint64_t min_dim, uint64_t max_dim) {
  std::uniform_int_distribution<uint64_t> dist(min_dim, max_dim);
  return dist(embag_rng);
}

// =============================================================================
// EmbagTestUtils Implementation
// =============================================================================

bool EmbagTestUtils::validate_embag_dimensions(uint64_t num_embeddings,
    uint64_t embedding_dim,
    uint64_t num_indices,
    uint64_t num_bags) {
  if (num_embeddings < 1 || embedding_dim < 1 || num_indices < 1) {
    return false;
  }
  // num_bags can be 0 for embedding lookup mode
  if (num_bags > num_indices) {
    return false;
  }
  return true;
}

data_type_t EmbagTestUtils::get_table_dtype(EmbagDataTypeCombination combo) {
  switch (combo) {
  case EmbagDataTypeCombination::F32_F32:
  case EmbagDataTypeCombination::F32_BF16:
    return data_type_t::f32;
  case EmbagDataTypeCombination::BF16_BF16:
  case EmbagDataTypeCombination::BF16_F32:
    return data_type_t::bf16;
  case EmbagDataTypeCombination::U4_F32:
  case EmbagDataTypeCombination::U4_BF16:
    return data_type_t::u4;
  case EmbagDataTypeCombination::S8_F32:
  case EmbagDataTypeCombination::S8_BF16:
    return data_type_t::s8;
  case EmbagDataTypeCombination::S4_F32:
  case EmbagDataTypeCombination::S4_BF16:
    return data_type_t::s4;
  default:
    return data_type_t::f32;
  }
}

data_type_t EmbagTestUtils::get_output_dtype(EmbagDataTypeCombination combo) {
  switch (combo) {
  case EmbagDataTypeCombination::F32_F32:
  case EmbagDataTypeCombination::BF16_F32:
  case EmbagDataTypeCombination::U4_F32:
  case EmbagDataTypeCombination::S8_F32:
  case EmbagDataTypeCombination::S4_F32:
    return data_type_t::f32;
  case EmbagDataTypeCombination::F32_BF16:
  case EmbagDataTypeCombination::BF16_BF16:
  case EmbagDataTypeCombination::U4_BF16:
  case EmbagDataTypeCombination::S8_BF16:
  case EmbagDataTypeCombination::S4_BF16:
    return data_type_t::bf16;
  default:
    return data_type_t::f32;
  }
}

bool EmbagTestUtils::is_valid_embag_data_type_combination(
  EmbagDataTypeCombination combo) {
  const auto &supported = EmbagParameterGenerator::supported_combinations;
  return std::find(supported.begin(), supported.end(), combo) != supported.end();
}

bool EmbagTestUtils::is_embag_kernel_supported(data_type_t table_dtype,
    data_type_t output_dtype,
    embag_algo_t algo) {
  // F32 table with F32 or BF16 output
  if (table_dtype == data_type_t::f32 &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    return true;
  }
  // BF16 table with BF16 or F32 output
  if (table_dtype == data_type_t::bf16 &&
      (output_dtype == data_type_t::bf16 || output_dtype == data_type_t::f32)) {
    return true;
  }
  // U4/S4/S8 quantized tables with F32 or BF16 output
  if ((table_dtype == data_type_t::u4 || table_dtype == data_type_t::s8 || 
       table_dtype == data_type_t::s4) &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    return true;
  }
  return false;
}

bool EmbagTestUtils::is_embag_reference_supported(data_type_t table_dtype,
    data_type_t output_dtype,
    embag_algo_t algo) {
  // Reference kernel supports F32 table with F32 or BF16 output
  if (table_dtype == data_type_t::f32 &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    return true;
  }
  // Reference supports BF16 table with BF16 or F32 output
  if (table_dtype == data_type_t::bf16 &&
      (output_dtype == data_type_t::bf16 || output_dtype == data_type_t::f32)) {
    return true;
  }
  // Reference supports U4/S4/S8 quantized with F32 or BF16 output
  if ((table_dtype == data_type_t::u4 || table_dtype == data_type_t::s8 || 
       table_dtype == data_type_t::s4) &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    return true;
  }
  return false;
}

std::vector<uint64_t> EmbagTestUtils::get_table_dims(uint64_t num_embeddings,
    uint64_t embedding_dim) {
  return {num_embeddings, embedding_dim};
}

std::vector<uint64_t> EmbagTestUtils::get_indices_dims(uint64_t num_indices) {
  return {num_indices};
}

std::vector<uint64_t> EmbagTestUtils::get_offsets_dims(uint64_t num_bags,
    bool include_last_offset) {
  return {include_last_offset ? num_bags + 1 : num_bags};
}

std::vector<uint64_t> EmbagTestUtils::get_output_dims_bag(uint64_t num_bags,
    uint64_t embedding_dim) {
  return {num_bags, embedding_dim};
}

std::vector<uint64_t> EmbagTestUtils::get_output_dims_lookup(
  uint64_t num_indices,
  uint64_t embedding_dim) {
  return {num_indices, embedding_dim};
}

std::vector<int64_t> EmbagTestUtils::generate_random_indices(
  uint64_t num_indices,
  uint64_t num_embeddings,
  int64_t padding_idx) {
  std::vector<int64_t> indices(num_indices);
  std::uniform_int_distribution<int64_t> dist(0, num_embeddings - 1);

  for (size_t i = 0; i < num_indices; ++i) {
    indices[i] = dist(embag_rng);
    // Optionally insert padding index
    if (padding_idx >= 0 && (embag_rng() % 10) == 0) {
      indices[i] = padding_idx;
    }
  }
  return indices;
}

std::vector<int64_t> EmbagTestUtils::generate_offsets(uint64_t num_bags,
    uint64_t num_indices,
    bool include_last_offset) {
  std::vector<int64_t> offsets;
  if (num_bags == 0) {
    return offsets;
  }

  offsets.push_back(0);
  uint64_t indices_per_bag = num_indices / num_bags;
  uint64_t remainder = num_indices % num_bags;

  for (uint64_t i = 1; i < num_bags; ++i) {
    uint64_t bag_size = indices_per_bag + (i <= remainder ? 1 : 0);
    offsets.push_back(offsets.back() + bag_size);
  }

  if (include_last_offset) {
    offsets.push_back(num_indices);
  }

  return offsets;
}

std::vector<int64_t> EmbagTestUtils::generate_boundary_indices(
  uint64_t num_indices,
  uint64_t num_embeddings) {
  std::vector<int64_t> indices(num_indices);
  // Fill with boundary values: 0, max, alternating
  for (size_t i = 0; i < num_indices; ++i) {
    if (i % 2 == 0) {
      indices[i] = 0;
    }
    else {
      indices[i] = num_embeddings - 1;
    }
  }
  return indices;
}

bool EmbagTestUtils::compare_embag_tensors(const tensor_t &test_tensor,
    const tensor_t &ref_tensor,
    float abs_tolerance,
    float rel_tolerance) {
  return AITestUtils::compare_sampled_tensors(test_tensor, ref_tensor,
         abs_tolerance, rel_tolerance);
}

status_t EmbagTestUtils::run_reference_embag(tensor_t &table,
    tensor_t &indices,
    tensor_t &offsets,
    tensor_t &output,
    const EmbagParamsAI &params) {
  try {
    using tensor_map_type = std::map<std::string, tensor_t>;
    tensor_map_type inputs;
    tensor_map_type outputs;

    inputs["indices"] = indices;
    if (params.use_offsets) {
      inputs["offsets"] = offsets;
    }
    outputs["output"] = output;

    // Create context
    tensor_t table_copy = table;
    table_copy.set_name("table");

    auto embag_context = embag_context_t()
                         .set_param("table", table_copy)
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
      return status_t::failure;
    }

    // Create operator
    auto embag_operator = embag_operator_t()
                          .set_name("embag_forced_ref_operator")
                          .set_context(embag_context)
                          .create();

    if (embag_operator.is_bad_object()) {
      return status_t::failure;
    }

    // Set inputs and outputs
    embag_operator = embag_operator.set_input("indices", indices);
    if (params.use_offsets) {
      embag_operator = embag_operator.set_input("offsets", offsets);
    }
    embag_operator = embag_operator.set_output("output", output);

    // Force reference kernel
    embag_operator = embag_operator.set_forced_kernel("reference");

    // Execute
    status_t status = embag_operator.execute();
    return status;
  }
  catch (const std::exception &e) {
    std::cerr << "[AI_EMBAG_REF] Exception in run_reference_embag: " << e.what()
              << std::endl;
    return status_t::failure;
  }
  catch (...) {
    std::cerr << "[AI_EMBAG_REF] Unknown exception in run_reference_embag" <<
              std::endl;
    return status_t::failure;
  }
}

float EmbagTestUtils::get_accuracy_tolerance(data_type_t output_dtype) {
  switch (output_dtype) {
  case data_type_t::f32:
    return AI_EMBAG_TOLERANCE_F32;
  case data_type_t::bf16:
    return AI_EMBAG_TOLERANCE_BF16;
  default:
    return AI_EMBAG_TOLERANCE_DEFAULT;
  }
}

float EmbagTestUtils::get_relative_tolerance(data_type_t output_dtype) {
  switch (output_dtype) {
  case data_type_t::f32:
    return AI_EMBAG_REL_TOLERANCE_F32;
  case data_type_t::bf16:
    return AI_EMBAG_REL_TOLERANCE_BF16;
  default:
    return AI_EMBAG_REL_TOLERANCE_DEFAULT;
  }
}

float EmbagTestUtils::get_epsilon_value(data_type_t dtype) {
  switch (dtype) {
  case data_type_t::f32:
    return AI_EMBAG_EPSILON_F32;
  case data_type_t::bf16:
    return AI_EMBAG_EPSILON_BF16;
  default:
    return AI_EMBAG_EPSILON_DEFAULT;
  }
}

// =============================================================================
// EmbagParameterGenerator Implementation
// =============================================================================

std::vector<EmbagDataTypeCombination>
EmbagParameterGenerator::supported_combinations = {
  EmbagDataTypeCombination::F32_F32,
  EmbagDataTypeCombination::F32_BF16,
  EmbagDataTypeCombination::BF16_BF16,
  EmbagDataTypeCombination::BF16_F32,
  EmbagDataTypeCombination::U4_F32,
  EmbagDataTypeCombination::U4_BF16,
  EmbagDataTypeCombination::S8_F32,
  EmbagDataTypeCombination::S8_BF16,
  EmbagDataTypeCombination::S4_F32,
  EmbagDataTypeCombination::S4_BF16,
};

std::vector<EmbagParamsAI>
EmbagParameterGenerator::generate_comprehensive_test_suite() {
  std::vector<EmbagParamsAI> all_params;
  add_accuracy_params(all_params);
  add_boundary_params(all_params);
  add_edge_case_params(all_params);
  add_invalid_params(all_params);
  add_embedding_lookup_params(all_params);
  return all_params;
}

std::vector<EmbagParamsAI>
EmbagParameterGenerator::generate_minimal_test_suite() {
  std::vector<EmbagParamsAI> minimal_params;
  add_minimal_accuracy_params(minimal_params);

  // Add minimal boundary test
  minimal_params.push_back(create_param(10, 8, 4, 2,
                                        EmbagDataTypeCombination::F32_F32,
                                        embag_algo_t::sum,
                                        TestCategory::BOUNDARY,
                                        true, true));

  // Add minimal invalid test
  minimal_params.push_back(create_param(0, 8, 4, 2,
                                        EmbagDataTypeCombination::F32_F32,
                                        embag_algo_t::sum,
                                        TestCategory::INVALID,
                                        true, false));

  return minimal_params;
}

std::vector<EmbagParamsAI>
EmbagParameterGenerator::generate_category_specific_params(
  TestCategory category) {
  std::vector<EmbagParamsAI> params;
  switch (category) {
  case TestCategory::ACCURACY:
    add_accuracy_params(params);
    break;
  case TestCategory::BOUNDARY:
    add_boundary_params(params);
    break;
  case TestCategory::EDGE_CASE:
    add_edge_case_params(params);
    add_embedding_lookup_params(params);
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

EmbagParamsAI
EmbagParameterGenerator::generate_random_params_for_accuracy_subcategory(
  const std::string &category,
  EmbagDataTypeCombination data_combo,
  embag_algo_t algo,
  bool expect_success) {

  uint64_t num_embeddings = 100, embedding_dim = 64;
  uint64_t num_indices = 16, num_bags = 4;

  if (category == "tiny") {
    num_embeddings = generate_random_embag_dim(
                       static_cast<uint64_t>(EmbagDimensions::TINY_EMBEDDINGS),
                       static_cast<uint64_t>(EmbagDimensions::TINY_EMBEDDINGS) * 2);
    embedding_dim = generate_random_embag_dim(
                      static_cast<uint64_t>(EmbagDimensions::TINY_DIM),
                      static_cast<uint64_t>(EmbagDimensions::TINY_DIM) * 2);
    num_indices = generate_random_embag_dim(
                    static_cast<uint64_t>(EmbagDimensions::TINY_INDICES),
                    static_cast<uint64_t>(EmbagDimensions::TINY_INDICES) * 2);
    num_bags = generate_random_embag_dim(
                 static_cast<uint64_t>(EmbagDimensions::TINY_BAGS),
                 static_cast<uint64_t>(EmbagDimensions::TINY_BAGS) * 2);
  }
  else if (category == "small") {
    num_embeddings = generate_random_embag_dim(
                       static_cast<uint64_t>(EmbagDimensions::SMALL_EMBEDDINGS),
                       static_cast<uint64_t>(EmbagDimensions::SMALL_EMBEDDINGS) * 2);
    embedding_dim = generate_random_embag_dim(
                      static_cast<uint64_t>(EmbagDimensions::SMALL_DIM),
                      static_cast<uint64_t>(EmbagDimensions::SMALL_DIM) * 2);
    num_indices = generate_random_embag_dim(
                    static_cast<uint64_t>(EmbagDimensions::SMALL_INDICES),
                    static_cast<uint64_t>(EmbagDimensions::SMALL_INDICES) * 2);
    num_bags = generate_random_embag_dim(
                 static_cast<uint64_t>(EmbagDimensions::SMALL_BAGS),
                 static_cast<uint64_t>(EmbagDimensions::SMALL_BAGS) * 2);
  }
  else if (category == "medium") {
    num_embeddings = generate_random_embag_dim(
                       static_cast<uint64_t>(EmbagDimensions::MEDIUM_EMBEDDINGS),
                       static_cast<uint64_t>(EmbagDimensions::MEDIUM_EMBEDDINGS) * 2);
    embedding_dim = generate_random_embag_dim(
                      static_cast<uint64_t>(EmbagDimensions::MEDIUM_DIM),
                      static_cast<uint64_t>(EmbagDimensions::MEDIUM_DIM) * 2);
    num_indices = generate_random_embag_dim(
                    static_cast<uint64_t>(EmbagDimensions::MEDIUM_INDICES),
                    static_cast<uint64_t>(EmbagDimensions::MEDIUM_INDICES) * 2);
    num_bags = generate_random_embag_dim(
                 static_cast<uint64_t>(EmbagDimensions::MEDIUM_BAGS),
                 static_cast<uint64_t>(EmbagDimensions::MEDIUM_BAGS) * 2);
  }
  else if (category == "large") {
    num_embeddings = generate_random_embag_dim(
                       static_cast<uint64_t>(EmbagDimensions::LARGE_EMBEDDINGS),
                       static_cast<uint64_t>(EmbagDimensions::LARGE_EMBEDDINGS));
    embedding_dim = generate_random_embag_dim(
                      static_cast<uint64_t>(EmbagDimensions::LARGE_DIM),
                      static_cast<uint64_t>(EmbagDimensions::LARGE_DIM) * 2);
    num_indices = generate_random_embag_dim(
                    static_cast<uint64_t>(EmbagDimensions::LARGE_INDICES),
                    static_cast<uint64_t>(EmbagDimensions::LARGE_INDICES));
    num_bags = generate_random_embag_dim(
                 static_cast<uint64_t>(EmbagDimensions::LARGE_BAGS),
                 static_cast<uint64_t>(EmbagDimensions::LARGE_BAGS));
  }

  // Ensure num_bags <= num_indices
  if (num_bags > num_indices) {
    num_bags = num_indices / 2;
    if (num_bags < 1) {
      num_bags = 1;
    }
  }

  return create_param(num_embeddings, embedding_dim, num_indices, num_bags,
                      data_combo, algo, TestCategory::ACCURACY, true, expect_success);
}

void EmbagParameterGenerator::add_minimal_accuracy_params(
  std::vector<EmbagParamsAI> &params) {
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
  fixed_dims = {
    {10, 8, 4, 2, "tiny"},
    {100, 64, 16, 4, "small"},
    {1000, 128, 64, 8, "medium"},
  };

  std::vector<embag_algo_t> algos = {
    embag_algo_t::sum,
    embag_algo_t::mean,
    embag_algo_t::max
  };

  for (auto data_combo : supported_combinations) {
    // Skip U4 for minimal tests - requires special quantized tensor handling
    if (data_combo == EmbagDataTypeCombination::U4_F32) {
      continue;
    }
    
    for (auto algo : algos) {
      for (const auto &[num_emb, emb_dim, num_idx, num_bag, desc] : fixed_dims) {
        if (EmbagTestUtils::is_embag_kernel_supported(
              EmbagTestUtils::get_table_dtype(data_combo),
              EmbagTestUtils::get_output_dtype(data_combo), algo)) {
          params.push_back(create_param(num_emb, emb_dim, num_idx, num_bag,
                                        data_combo, algo, TestCategory::ACCURACY, true, true));
        }
      }
    }
  }
}

void EmbagParameterGenerator::add_accuracy_params(
  std::vector<EmbagParamsAI> &params) {
  const std::vector<std::string> categories = {"tiny", "small", "medium",
                                                "large"
                                               };
  const int max_cases_per_category = 5;

  std::vector<embag_algo_t> algos = {
    embag_algo_t::sum,
    embag_algo_t::mean,
    embag_algo_t::max
  };

  for (const auto &category : categories) {
    for (auto data_combo : supported_combinations) {
      for (auto algo : algos) {
        if (EmbagTestUtils::is_embag_kernel_supported(
              EmbagTestUtils::get_table_dtype(data_combo),
              EmbagTestUtils::get_output_dtype(data_combo), algo)) {
          for (int i = 0; i < max_cases_per_category; i++) {
            params.push_back(generate_random_params_for_accuracy_subcategory(
                               category, data_combo, algo, true));
          }
        }
      }
    }
  }
}

void EmbagParameterGenerator::add_boundary_params(
  std::vector<EmbagParamsAI> &params) {
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
  boundary_dims = {
    {10, 8, 1, 1, "minimal_indices"},
    {10, 8, 4, 1, "single_bag"},
    {10, 1, 4, 2, "minimal_dim"},
    {1000, 512, 64, 8, "large_dim"},
  };

  std::vector<embag_algo_t> algos = {embag_algo_t::sum, embag_algo_t::mean};

  for (auto data_combo : supported_combinations) {
    for (auto algo : algos) {
      for (const auto &[num_emb, emb_dim, num_idx, num_bag, desc] :
           boundary_dims) {
        if (EmbagTestUtils::is_embag_kernel_supported(
              EmbagTestUtils::get_table_dtype(data_combo),
              EmbagTestUtils::get_output_dtype(data_combo), algo)) {
          params.push_back(create_param(num_emb, emb_dim, num_idx, num_bag,
                                        data_combo, algo, TestCategory::BOUNDARY, true, true));
        }
      }
    }
  }
}

void EmbagParameterGenerator::add_edge_case_params(
  std::vector<EmbagParamsAI> &params) {
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>> edge_dims = {
    {10000, 256, 256, 32},  // Large table
    {100, 512, 64, 8},      // Large embedding dim
    {1000, 64, 1, 1},       // Single index
  };

  for (auto data_combo : supported_combinations) {
    for (const auto &[num_emb, emb_dim, num_idx, num_bag] : edge_dims) {
      if (EmbagTestUtils::is_embag_kernel_supported(
            EmbagTestUtils::get_table_dtype(data_combo),
            EmbagTestUtils::get_output_dtype(data_combo), embag_algo_t::sum)) {
        params.push_back(create_param(num_emb, emb_dim, num_idx, num_bag,
                                      data_combo, embag_algo_t::sum, TestCategory::EDGE_CASE, true, true));
      }
    }
  }
}

void EmbagParameterGenerator::add_invalid_params(
  std::vector<EmbagParamsAI> &params) {
  for (auto data_combo : supported_combinations) {
    // Invalid dimensions - zero values
    params.push_back(create_param(0, 64, 16, 4, data_combo, embag_algo_t::sum,
                                  TestCategory::INVALID, true, false));
    params.push_back(create_param(100, 0, 16, 4, data_combo, embag_algo_t::sum,
                                  TestCategory::INVALID, true, false));
    params.push_back(create_param(100, 64, 0, 4, data_combo, embag_algo_t::sum,
                                  TestCategory::INVALID, true, false));
    
    // Note: num_bags > num_indices is NOT tested as invalid because the operator
    // allows this configuration (creates empty bags with zero output)
  }
}

void EmbagParameterGenerator::add_embedding_lookup_params(
  std::vector<EmbagParamsAI> &params) {
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> lookup_dims = {
    {100, 64, 16},
    {1000, 128, 64},
    {10000, 256, 128},
  };

  for (auto data_combo : supported_combinations) {
    for (const auto &[num_emb, emb_dim, num_idx] : lookup_dims) {
      if (EmbagTestUtils::is_embag_kernel_supported(
            EmbagTestUtils::get_table_dtype(data_combo),
            EmbagTestUtils::get_output_dtype(data_combo), embag_algo_t::none)) {
        params.push_back(create_param(num_emb, emb_dim, num_idx, 0,
                                      data_combo, embag_algo_t::none, TestCategory::EDGE_CASE, false, true));
      }
    }
  }
}

void EmbagParameterGenerator::generate_reference_kernel_exhaustive_params(
  std::vector<EmbagParamsAI> &params) {
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string>>
  ref_dims = {
    {10, 8, 4, 2, "tiny"},
    {100, 64, 16, 4, "small"},
    {1000, 128, 64, 8, "medium"},
    {10000, 256, 128, 16, "large"},
  };

  std::vector<embag_algo_t> algos = {
    embag_algo_t::sum,
    embag_algo_t::mean,
    embag_algo_t::max,
    embag_algo_t::none
  };

  for (auto data_combo : supported_combinations) {
    for (auto algo : algos) {
      for (const auto &[num_emb, emb_dim, num_idx, num_bag, desc] : ref_dims) {
        if (EmbagTestUtils::is_embag_reference_supported(
              EmbagTestUtils::get_table_dtype(data_combo),
              EmbagTestUtils::get_output_dtype(data_combo), algo)) {
          bool use_offsets = (algo != embag_algo_t::none);
          params.push_back(create_param(num_emb, emb_dim, num_idx,
                                        use_offsets ? num_bag : 0,
                                        data_combo, algo, TestCategory::REFERENCE_KERNEL,
                                        use_offsets, true));
        }
      }
    }
  }
}

EmbagParamsAI EmbagParameterGenerator::create_param(
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices, uint64_t num_bags,
  EmbagDataTypeCombination data_types,
  embag_algo_t algo,
  TestCategory category,
  bool use_offsets,
  bool expect_success) {

  EmbagParamsAI param;
  param.num_embeddings = num_embeddings;
  param.embedding_dim = embedding_dim;
  param.num_indices = num_indices;
  param.num_bags = num_bags;
  param.data_types = data_types;
  param.algo = algo;
  param.category = category;
  param.use_offsets = use_offsets;
  param.expect_success = expect_success;
  
  // Set fp16_scale_bias to true for all quantized embeddings (U4/S8/S4)
  if (data_types == EmbagDataTypeCombination::U4_F32 || 
      data_types == EmbagDataTypeCombination::U4_BF16 ||
      data_types == EmbagDataTypeCombination::S8_F32 ||
      data_types == EmbagDataTypeCombination::S8_BF16 ||
      data_types == EmbagDataTypeCombination::S4_F32 ||
      data_types == EmbagDataTypeCombination::S4_BF16) {
    param.fp16_scale_bias = true;
  }

  static std::atomic<uint64_t> param_counter{0};

  // Build test name
  auto dtype_to_str = [](data_type_t dt) {
    switch (dt) {
    case data_type_t::f32:
      return "f32";
    case data_type_t::bf16:
      return "bf16";
    case data_type_t::u4:
      return "u4";
    case data_type_t::s8:
      return "s8";
    case data_type_t::s4:
      return "s4";
    default:
      return "unk";
    }
  };

  auto algo_to_str = [](embag_algo_t a) {
    switch (a) {
    case embag_algo_t::sum:
      return "sum";
    case embag_algo_t::mean:
      return "mean";
    case embag_algo_t::max:
      return "max";
    case embag_algo_t::none:
      return "lookup";
    default:
      return "unk";
    }
  };

  std::string table_dtype_str = dtype_to_str(
                                  EmbagTestUtils::get_table_dtype(data_types));
  std::string output_dtype_str = dtype_to_str(
                                   EmbagTestUtils::get_output_dtype(data_types));
  std::string algo_str = algo_to_str(algo);

  param.test_name = "emb" + std::to_string(num_embeddings) +
                    "_dim" + std::to_string(embedding_dim) +
                    "_idx" + std::to_string(num_indices) +
                    "_bag" + std::to_string(num_bags) +
                    "_tbl_" + table_dtype_str +
                    "_out_" + output_dtype_str +
                    "_" + algo_str +
                    "_" + std::to_string(param_counter.fetch_add(1));

  return param;
}

} // namespace ai_gtests
