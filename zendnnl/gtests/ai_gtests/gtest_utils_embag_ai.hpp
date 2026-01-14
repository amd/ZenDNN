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

#ifndef _GTEST_UTILS_EMBAG_AI_HPP_
#define _GTEST_UTILS_EMBAG_AI_HPP_

#include "gtest_utils_ai.hpp"
#include "operators/embag/embag_context.hpp"
#include "operators/embag/embag_operator.hpp"

namespace ai_gtests {

using namespace zendnnl::ops;

// Accuracy tolerance macros for embedding bag output data types
#define AI_EMBAG_TOLERANCE_F32 1e-4f
#define AI_EMBAG_TOLERANCE_BF16 1e-2f
#define AI_EMBAG_TOLERANCE_U4 1e-2f
#define AI_EMBAG_TOLERANCE_DEFAULT 1e-5f

#define AI_EMBAG_REL_TOLERANCE_F32 1e-5f
#define AI_EMBAG_REL_TOLERANCE_BF16 1e-2f
#define AI_EMBAG_REL_TOLERANCE_U4 1e-2f
#define AI_EMBAG_REL_TOLERANCE_DEFAULT 1e-5f

#define AI_EMBAG_EPSILON_F32     1.19e-7
#define AI_EMBAG_EPSILON_BF16    9.76e-4
#define AI_EMBAG_EPSILON_U4      9.76e-4
#define AI_EMBAG_EPSILON_DEFAULT 9.76e-4

/** @brief Embedding table dimension categories */
enum class EmbagDimensions : uint64_t {
  // Embedding table sizes (num_embeddings)
  TINY_EMBEDDINGS = 10,
  SMALL_EMBEDDINGS = 100,
  MEDIUM_EMBEDDINGS = 1000,
  LARGE_EMBEDDINGS = 10000,
  
  // Embedding dimensions
  TINY_DIM = 8,
  SMALL_DIM = 64,
  MEDIUM_DIM = 128,
  LARGE_DIM = 256,
  XLARGE_DIM = 512,
  
  // Number of indices
  TINY_INDICES = 4,
  SMALL_INDICES = 16,
  MEDIUM_INDICES = 64,
  LARGE_INDICES = 256,
  
  // Number of bags
  TINY_BAGS = 2,
  SMALL_BAGS = 8,
  MEDIUM_BAGS = 32,
  LARGE_BAGS = 128,
  
  // Range values for random generation
  MIN_EMBEDDINGS = 10,
  MAX_EMBEDDINGS = 10000,
  MIN_DIM = 8,
  MAX_DIM = 512,
  MIN_INDICES = 1,
  MAX_INDICES = 1000,
  MIN_BAGS = 1,
  MAX_BAGS = 256
};

/** @brief Data type combinations for embedding bag */
enum class EmbagDataTypeCombination {
  F32_F32,      // Table: F32, Output: F32
  F32_BF16,     // Table: F32, Output: BF16
  BF16_BF16,    // Table: BF16, Output: BF16
  BF16_F32,     // Table: BF16, Output: F32
  U4_F32,       // Table: U4 (quantized), Output: F32
  U4_BF16,      // Table: U4 (quantized), Output: BF16
  S8_F32,       // Table: S8 (quantized), Output: F32
  S8_BF16,      // Table: S8 (quantized), Output: BF16
  S4_F32,       // Table: S4 (quantized), Output: F32
  S4_BF16,      // Table: S4 (quantized), Output: BF16
};

/** @brief AI-specific embedding bag parameter structure */
struct EmbagParamsAI {
  uint64_t num_embeddings;  // Number of embeddings in table
  uint64_t embedding_dim;   // Dimension of each embedding
  uint64_t num_indices;     // Total number of indices
  uint64_t num_bags;        // Number of bags (for embedding bag mode)
  
  EmbagDataTypeCombination data_types;
  embag_algo_t algo;        // sum, mean, max, or none (for embedding lookup)
  TestCategory category;
  
  bool use_offsets;         // If false, embedding lookup mode (no aggregation)
  bool use_padding_idx;     // Whether to use padding index
  int64_t padding_idx;      // Padding index value
  bool fp16_scale_bias;     // For quantized embeddings (U4/S8)
  bool include_last_offset; // Include last offset in offsets tensor
  
  bool expect_success;
  std::string test_name;

  EmbagParamsAI() : num_embeddings(100), embedding_dim(64), 
    num_indices(16), num_bags(4),
    data_types(EmbagDataTypeCombination::F32_F32),
    algo(embag_algo_t::sum),
    category(TestCategory::ACCURACY),
    use_offsets(true),
    use_padding_idx(false),
    padding_idx(-1),
    fp16_scale_bias(false),
    include_last_offset(false),
    expect_success(true),
    test_name("") {}
};

/** @brief Embedding bag-specific utility functions */
class EmbagTestUtils {
 public:
  // Dimension validation
  static bool validate_embag_dimensions(uint64_t num_embeddings, 
                                        uint64_t embedding_dim,
                                        uint64_t num_indices, 
                                        uint64_t num_bags);
  
  // Data type utilities
  static data_type_t get_table_dtype(EmbagDataTypeCombination combo);
  static data_type_t get_output_dtype(EmbagDataTypeCombination combo);
  static bool is_valid_embag_data_type_combination(EmbagDataTypeCombination combo);
  
  // Kernel support utilities
  static bool is_embag_kernel_supported(data_type_t table_dtype,
                                        data_type_t output_dtype,
                                        embag_algo_t algo);
  static bool is_embag_reference_supported(data_type_t table_dtype,
                                           data_type_t output_dtype,
                                           embag_algo_t algo);
  
  // Tensor dimension helpers
  static std::vector<uint64_t> get_table_dims(uint64_t num_embeddings, 
                                              uint64_t embedding_dim);
  static std::vector<uint64_t> get_indices_dims(uint64_t num_indices);
  static std::vector<uint64_t> get_offsets_dims(uint64_t num_bags, 
                                                bool include_last_offset);
  static std::vector<uint64_t> get_output_dims_bag(uint64_t num_bags, 
                                                   uint64_t embedding_dim);
  static std::vector<uint64_t> get_output_dims_lookup(uint64_t num_indices, 
                                                      uint64_t embedding_dim);
  
  // Index and offset generation
  static std::vector<int64_t> generate_random_indices(uint64_t num_indices,
                                                      uint64_t num_embeddings,
                                                      int64_t padding_idx = -1);
  static std::vector<int64_t> generate_offsets(uint64_t num_bags,
                                               uint64_t num_indices,
                                               bool include_last_offset = false);
  static std::vector<int64_t> generate_boundary_indices(uint64_t num_indices,
                                                        uint64_t num_embeddings);
  
  // Tensor comparison
  static bool compare_embag_tensors(const tensor_t &test_tensor,
                                    const tensor_t &ref_tensor,
                                    float abs_tolerance,
                                    float rel_tolerance);
  
  // Reference implementation
  static status_t run_reference_embag(tensor_t &table,
                                      tensor_t &indices,
                                      tensor_t &offsets,
                                      tensor_t &output,
                                      const EmbagParamsAI &params);
  
  // Tolerance helpers
  static float get_accuracy_tolerance(data_type_t output_dtype);
  static float get_relative_tolerance(data_type_t output_dtype);
  static float get_epsilon_value(data_type_t dtype);
};

/** @brief Embedding bag parameter generator */
class EmbagParameterGenerator {
 public:
  static std::vector<EmbagDataTypeCombination> supported_combinations;
  
  static std::vector<EmbagParamsAI> generate_comprehensive_test_suite();
  static std::vector<EmbagParamsAI> generate_minimal_test_suite();
  static std::vector<EmbagParamsAI> generate_category_specific_params(
    TestCategory category);

 private:
  static EmbagParamsAI generate_random_params_for_accuracy_subcategory(
    const std::string &category,
    EmbagDataTypeCombination data_combo,
    embag_algo_t algo,
    bool expect_success);
  
  static void add_minimal_accuracy_params(std::vector<EmbagParamsAI> &params);
  static void add_accuracy_params(std::vector<EmbagParamsAI> &params);
  static void add_boundary_params(std::vector<EmbagParamsAI> &params);
  static void add_edge_case_params(std::vector<EmbagParamsAI> &params);
  static void add_invalid_params(std::vector<EmbagParamsAI> &params);
  static void add_embedding_lookup_params(std::vector<EmbagParamsAI> &params);
  static void generate_reference_kernel_exhaustive_params(
    std::vector<EmbagParamsAI> &params);
  
  static EmbagParamsAI create_param(
    uint64_t num_embeddings, uint64_t embedding_dim,
    uint64_t num_indices, uint64_t num_bags,
    EmbagDataTypeCombination data_types,
    embag_algo_t algo,
    TestCategory category,
    bool use_offsets = true,
    bool expect_success = true);
};

} // namespace ai_gtests

#endif // _GTEST_UTILS_EMBAG_AI_HPP_
