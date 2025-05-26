/********************************************************************************
# * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// To pass command line arguments to gtest
int gtest_argc;
char **gtest_argv;

const uint32_t po_size = 6; //Supported postop
vector<post_op_type_t> po_arr;
unordered_map<std::string, int> po_map;

//Matmul Tolerance Limit
const float MATMUL_F32_TOL = 0.001;
const float MATMUL_BF16_TOL = 0.1;

//number of testcases and random seed
const uint32_t TEST_NUM = 100; //ToDo: make it command line argument
int seed = time(NULL);

/** @brief matmul_test Data Structure(vector of structures) to hold random Matmul Parameters */
std::vector<MatmulType> matmul_test{};

int main(int argc, char **argv) {
  //ToDO: Write a Command-line parser to avoid hardcodings in cmd arguments
  //Supported Postop
  po_arr = { post_op_type_t::relu, post_op_type_t::gelu_tanh,
             post_op_type_t::gelu_erf, post_op_type_t::sigmoid,
             post_op_type_t::swish, post_op_type_t::tanh
           };

  //Postop string to index
  po_map["relu"]=0;
  po_map["gelu_tanh"]=1;
  po_map["gelu_erf"]=2;
  po_map["sigmoid"]=3;
  po_map["swish"]=4;
  po_map["tanh"]=5;

  //If Seed is provided as command line argument
  if (argc==3) {
    try {
      seed = stoi(argv[2]);
    }
    catch (const invalid_argument &e) {
      log_verbose("Invalid argument(using default seed): ", e.what());
    }
  }
  srand(seed);
  log_info("Value ", seed, " is used as seed. ");

  //Creating Random parameters for Matmul
  matmul_test.resize(TEST_NUM);

  testing :: InitGoogleTest(&argc, argv);
  gtest_argc = argc;
  gtest_argv = argv;
  return RUN_ALL_TESTS();
}
