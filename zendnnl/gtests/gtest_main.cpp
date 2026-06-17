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
#include "gtest_utils.hpp"
#include "ai_gtests/gtest_utils_ai.hpp"
// Group-matmul-specific declarations (`GroupQuantMatmulType`,
// `quant_matmul_test`) live under `group_matmul/`.  This is the only
// outside-of-the-subdir reference; the subdir's CMakeLists.txt + its
// helper TU own everything else.
#include "group_matmul/group_matmul_test_helpers.hpp"
// Reorder fixture param (`ReorderType`), the `reorder_test` global, and the
// reorder kernel/compare helpers live in the reorder/ subfolder.
#include "reorder/reorder_test_helpers.hpp"
// Group-embag-specific declarations (`GroupEmbagType`,
// `group_embag_test`) live under `group_embag/`.  Same pattern as
// `group_matmul/`: extern in the helpers header, definition + resize
// loop in this TU.
#include "group_embag/group_embag_test_helpers.hpp"
#include <ctime>
#include <unordered_set>

using namespace std;

const uint32_t po_size = 10; //Supported postops

const uint32_t dtype_size = 5; //Supported dtypes
vector<data_type_t> dtype_arr(dtype_size);


// Matmul Tolerance Limit
// TODO: Make the tolerance value dynamic based on factors such
// as tensor dimensions, data type, and value range.
const float REORDER_TOL     = 0.0001;
const float MATMUL_F32_TOL  = 0.001;
const float MATMUL_BF16_TOL = 0.01;
const float epsilon_f32     = 1.19e-7;
const float rtol_f32        = 1e-5;
const float epsilon_bf16    = 9.76e-4;
const float rtol_bf16       = 1e-2;
const float epsilon_woq     = 1.19e-5;
const float rtol_woq        = 1e-4;

// Matmul fused post-op parameters (LOWOHA `matmul_post_op` and reference `post_op_t`;
// not GEMM alpha/beta). clip uses (CLIP_LOWER, CLIP_UPPER) as (lower, upper) bounds.
const float MATMUL_POSTOP_CLIP_LOWER    = -0.5f;
const float MATMUL_POSTOP_CLIP_UPPER    =  0.5f;
const float MATMUL_POSTOP_ELTWISE_ALPHA = 1.0f;

// Test tolerance constants
const float EMBAG_F32_TOL  = 0.001;
const float EMBAG_BF16_TOL = 0.01;
const float EMBAG_F16_TOL  = 0.01;
const float EMBAG_INT4_TOL = 0.01;

// LOWOHA Reorder tolerance constants
const float LOWOHA_REORDER_INT8_TOL = 0.01;   // For int8/uint8 output
const float LOWOHA_REORDER_BF16_TOL = 0.01;  // For bf16 output
const float LOWOHA_REORDER_F32_TOL  = 0.001; // For f32 output

// Normalization tolerance constants
const float NORM_F32_TOL  = 0.001;
const float NORM_BF16_TOL = 0.01;
const float NORM_F16_TOL  = 0.01;

// Softmax tolerance constants
const float SOFTMAX_F32_TOL  = 0.001;
const float SOFTMAX_BF16_TOL = 0.01;
const float SOFTMAX_F16_TOL  = 0.01;

//number of testcases, random seed and empty post_op
uint32_t test_num      = 400;
int64_t  seed          = static_cast<int64_t>(std::time(nullptr));
std::string cmd_lowoha {};
uint32_t cmd_num_threads = 0;
std::string cmd_input_file {};
std::string cmd_operator {};
uint32_t ndims = 2;
std::string ai_test_mode_str {};
CLIParams cli_params {};

/** @brief matmul_test Data Structure(vector of structures) to hold random Matmul Parameters */
std::vector<MatmulType> matmul_test{};

/** @brief quant_matmul_test holds the constrained-by-construction parameters
 *  for the quantized group-matmul gtest fixture (TestGroupMatmulQuant)*/
std::vector<GroupQuantMatmulType> quant_matmul_test{};

/** @brief batchmatmul_test Data Structure(vector of structures) to hold random BatchMatmul Parameters */
std::vector<BatchMatmulType> batchmatmul_test{};

/** @brief reorder_test Data Structure(vector of structures) to hold random Reorder Parameters */
std::vector<ReorderType> reorder_test{};

/** @brief embag_test Data Structure(vector of structures) to hold random Embedding Bag Parameters */
std::vector<EmbagType> embag_test{};

/** @brief embedding_test Data Structure(vector of structures) to hold random Embedding Parameters */
std::vector<EmbeddingType> embedding_test{};

/** @brief group_embag_test holds random parameters for the
 *  TestGroupEmbag* fixtures in `group_embag/`. */
std::vector<GroupEmbagType> group_embag_test{};

/** @brief normalization_test Data Structure(vector of structures) to hold random Normalization Parameters */
std::vector<NormalizationType> normalization_test{};

/** @brief sdpa_test Data Structure(vector of structures) to hold random SDPA Parameters */
std::vector<SdpaType> sdpa_test{};

/** @brief softmax_test Data Structure(vector of structures) to hold random Softmax Parameters */
std::vector<SoftmaxType> softmax_test{};

int main(int argc, char **argv) {
  try {
    // Load global config (env/defaults) before tests read it.
    (void)zendnnl::common::zendnnl_global_block();
    dtype_arr = {data_type_t::f32, data_type_t::bf16, data_type_t::s8, data_type_t::u8, data_type_t::f16};
    // Command line argument parser
    Parser parse;
    parse(argc, argv, seed, test_num, ai_test_mode_str,
          cmd_lowoha,
          cmd_num_threads, cmd_input_file, cmd_operator, ndims,
          cli_params);

    static const std::unordered_set<std::string> k_input_file_ops = {
      "matmul", "reorder", "embeddingbag", "embedding", "normalization"
    };
    if (!cmd_input_file.empty() && cmd_operator.empty()) {
      commonlog_error("--input_file requires --op "
                      "(matmul, reorder, embeddingbag, embedding, normalization).\n");
      return 1;
    }
    if (!cmd_operator.empty()
        && k_input_file_ops.find(cmd_operator) == k_input_file_ops.end()) {
      commonlog_error("Invalid --op \"" + cmd_operator + "\". "
                      "Supported: matmul, reorder, embeddingbag, embedding, "
                      "normalization.\n");
      return 1;
    }
    if (!cmd_operator.empty() && cmd_input_file.empty()) {
      commonlog_error("--op requires --input_file.\n");
      return 1;
    }
    if (!cmd_input_file.empty() && cmd_operator == "matmul"
        && ndims != 2 && ndims != 3) {
      commonlog_error("--op matmul with --input_file requires --ndims 2 or 3 "
                      "(got " + std::to_string(ndims) + ").\n");
      return 1;
    }
    if (!cmd_input_file.empty() && cmd_operator == "reorder") {
      if (cmd_lowoha.empty()) {
        commonlog_error("Reorder input file requires --lowoha true or --lowoha false "
                        "(use separate files for regular vs LOWOHA suites).\n");
        return 1;
      }
      if (!parse_bool_field(cmd_lowoha, "lowoha").has_value()) {
        return 1;
      }
    }

    // Initialize AI test mode from command-line argument
    ai_gtests::initialize_test_mode(ai_test_mode_str);

    // Initialize AI LOWOHA mode from command-line argument
    ai_gtests::initialize_lowoha_mode(cmd_lowoha);

    srand(static_cast<unsigned int>(seed));
    std::cout << "Value " << seed << " is used as seed. \n";

    // --input_file and --op are paired; only populate the matching parameter vector.
    const bool input_file_mode = !cmd_input_file.empty();

    if (input_file_mode && cmd_operator == "matmul" && ndims == 2) {
      std::cout << "Using input file: " << cmd_input_file << std::endl;
      auto matmul_inputs = read_matmul_inputs(cmd_input_file, ndims);
      if (matmul_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      matmul_test.reserve(matmul_inputs.size() * test_num);
      for (const auto &input : matmul_inputs) {
        try {
          for (uint32_t i = 0; i < test_num; ++i) {
            matmul_test.push_back(MatmulType(input, i, test_num));
          }
        }
        catch (const std::exception &e) {
          commonlog_error(e.what());
        }
      }
      if (matmul_test.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
    }
    else if (!input_file_mode) {
      // Creating Random parameters for Matmul
      matmul_test.resize(test_num);
      for (uint32_t i = 0; i < test_num; ++i) {
        matmul_test[i] = MatmulType(cli_params.matmul_input, i, test_num);
      }
    }
    if (!input_file_mode) {
      // Constrained-parameter set for the quantized group-matmul fixture.
      quant_matmul_test.resize(test_num);
      for (uint32_t i = 0; i < test_num; ++i) {
        quant_matmul_test[i] = GroupQuantMatmulType(i, test_num);
      }
    }
    if (input_file_mode && cmd_operator == "matmul" && ndims == 3) {
      std::cout << "Using input file: " << cmd_input_file << std::endl;
      auto matmul_inputs = read_matmul_inputs(cmd_input_file, ndims);
      if (matmul_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      batchmatmul_test.reserve(matmul_inputs.size() * test_num);
      for (const auto &input : matmul_inputs) {
        try {
          for (uint32_t i = 0; i < test_num; ++i) {
            batchmatmul_test.push_back(BatchMatmulType(input, i, test_num));
          }
        }
        catch (const std::exception &e) {
          commonlog_error(e.what());
        }
      }
      if (batchmatmul_test.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
    }
    else if (!input_file_mode) {
      // Creating Random parameters for BatchMatmul
      batchmatmul_test.resize(test_num);
      for (uint32_t i = 0; i < test_num; ++i) {
        batchmatmul_test[i] = BatchMatmulType(cli_params.matmul_input, i, test_num);
      }
    }
    if (input_file_mode && cmd_operator == "reorder") {
      std::cout << "Using input file: " << cmd_input_file
                << " (lowoha=" << cmd_lowoha << ")" << std::endl;
      const bool is_lowoha_reorder = *parse_bool_field(cmd_lowoha, "lowoha");
      auto reorder_inputs = read_reorder_inputs(cmd_input_file, is_lowoha_reorder);
      if (reorder_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      reorder_test.reserve(reorder_inputs.size() * test_num);
      for (const auto &input : reorder_inputs) {
        try {
          for (uint32_t i = 0; i < test_num; ++i) {
            reorder_test.push_back(ReorderType(input, i, test_num));
          }
        }
        catch (const std::exception &e) {
          commonlog_error(e.what());
        }
      }
      if (reorder_test.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
    }
    else if (!input_file_mode) {
      // Create reorder tests: --lowoha true = LOWOHA tests, --lowoha false = regular, omit = randomised.
      for (uint32_t i = 0; i < test_num; ++i) {
        reorder_test.push_back(ReorderType(cli_params.reorder_input, i, test_num));
      }
    }
    if (input_file_mode && cmd_operator == "embeddingbag") {
      std::cout << "Using input file: " << cmd_input_file << std::endl;
      auto embag_inputs = read_embag_inputs(cmd_input_file);
      if (embag_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      embag_test.reserve(embag_inputs.size() * test_num);
      for (const auto &input : embag_inputs) {
        for (uint32_t i = 0; i < test_num; ++i) {
          embag_test.push_back(EmbagType(input));
        }
      }
    }
    else if (!input_file_mode) {
      // Creating Random parameters for Embedding Bag
      for (uint32_t i = 0; i < test_num; ++i) {
        embag_test.push_back(EmbagType(cli_params.embag_input));
      }
    }
    if (input_file_mode && cmd_operator == "embedding") {
      std::cout << "Using input file: " << cmd_input_file << std::endl;
      auto embedding_inputs = read_embedding_inputs(cmd_input_file);
      if (embedding_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      embedding_test.reserve(embedding_inputs.size() * test_num);
      for (const auto &input : embedding_inputs) {
        for (uint32_t i = 0; i < test_num; ++i) {
          embedding_test.push_back(EmbeddingType(input));
        }
      }
    }
    else if (!input_file_mode) {
      // Creating Random parameters for Embedding
      for (uint32_t i = 0; i < test_num; ++i) {
        embedding_test.push_back(EmbeddingType(cli_params.embedding_input));
      }
    }

    if (!input_file_mode) {
      // Creating Random parameters for Group Embedding Bag
      group_embag_test.resize(test_num);
      for (uint32_t i = 0; i < test_num; ++i) {
        group_embag_test[i] = GroupEmbagType(i, test_num);
      }
    }

    if (input_file_mode && cmd_operator == "normalization") {
      std::cout << "Using input file: " << cmd_input_file << std::endl;
      auto norm_inputs = read_normalization_inputs(cmd_input_file);
      if (norm_inputs.empty()) {
        commonlog_error("No valid test inputs parsed from --input_file \""
                        + cmd_input_file
                        + "\". Check the file path and line format.\n");
        return 1;
      }
      normalization_test.reserve(norm_inputs.size() * test_num);
      for (const auto &input : norm_inputs) {
        for (uint32_t i = 0; i < test_num; ++i) {
          normalization_test.push_back(NormalizationType(input));
        }
      }
    }
    else if (!input_file_mode) {
      // Creating Random parameters for Normalization
      for (uint32_t i = 0; i < test_num; ++i) {
        normalization_test.push_back(NormalizationType(cli_params.normalization_input));
      }
    }

    if (!input_file_mode) {
      // Creating Random parameters for SDPA
      sdpa_test.resize(test_num);

      // Creating Random parameters for Softmax
      softmax_test.resize(test_num);
    }

    ::testing :: InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
  catch (const std::exception &e) {
    std::cerr << "Exception caught in main: " << e.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown exception caught in main" << std::endl;
    return 1;
  }
}
