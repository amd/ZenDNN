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

#include <gtest/gtest.h>
#include "gtest_utils.hpp"

using namespace std;

const uint32_t po_size = 8; //Supported postop
vector<std::pair<std::string, post_op_type_t>> po_arr(po_size);

const uint32_t dtype_size = 4; //Supported dtype
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

// Test tolerance constants
const float EMBAG_F32_TOL  = 0.001;
const float EMBAG_BF16_TOL = 0.01;

//number of testcases, random seed and empty post_op
uint32_t test_num      = 400;
int seed               = time(NULL);
std::string cmd_post_op {};

/** @brief matmul_test Data Structure(vector of structures) to hold random Matmul Parameters */
std::vector<MatmulType> matmul_test{};

/** @brief batchmatmul_test Data Structure(vector of structures) to hold random BatchMatmul Parameters */
std::vector<BatchMatmulType> batchmatmul_test{};

/** @brief reorder_test Data Structure(vector of structures) to hold random Reorder Parameters */
std::vector<ReorderType> reorder_test{};

/** @brief embag_test Data Structure(vector of structures) to hold random Embedding Bag Parameters */
std::vector<EmbagType> embag_test{};

/** @brief embedding_test Data Structure(vector of structures) to hold random Embedding Parameters */
std::vector<EmbeddingType> embedding_test{};

int main(int argc, char **argv) {
  //Supported Postop
  po_arr = { {"relu", post_op_type_t::relu},
    {"gelu_tanh", post_op_type_t::gelu_tanh},
    {"gelu_erf", post_op_type_t::gelu_erf},
    {"sigmoid", post_op_type_t::sigmoid},
    {"swish", post_op_type_t::swish},
    {"tanh", post_op_type_t::tanh},
    {"binary_add", post_op_type_t::binary_add},
    {"binary_mul", post_op_type_t::binary_mul}
  };

  dtype_arr = {data_type_t::f32, data_type_t::bf16, data_type_t::s8, data_type_t::u8};
  // Command line argument parser
  Parser parse;
  parse(argc, argv, seed, test_num, cmd_post_op);
  srand(seed);
  std::cout<<"Value "<<seed<<" is used as seed. \n";

  // Creating Random parameters for Matmul
  matmul_test.resize(test_num);
  for (uint32_t i = 0; i < test_num; ++i) {
    matmul_test[i] = MatmulType(i, test_num);
  }
  // Creating Random parameters for BatchMatmul
  batchmatmul_test.resize(test_num);
  for (uint32_t i = 0; i < test_num; ++i) {
    batchmatmul_test[i] = BatchMatmulType(i, test_num);
  }
  // Creating Random parameters for Reorder
  reorder_test.resize(test_num);
  for (uint32_t i = 0; i < test_num; ++i) {
    reorder_test[i] = ReorderType(i, test_num);
  }
  // Creating Random parameters for Embedding Bag
  embag_test.resize(test_num);
  // Creating Random parameters for Embedding
  embedding_test.resize(test_num);

  ::testing :: InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
