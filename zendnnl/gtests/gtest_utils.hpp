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

#ifndef _GTEST_UTILS_HPP_
#define _GTEST_UTILS_HPP_
#include <string>
#include <random>
#include <algorithm>
#include <variant>
#include <omp.h>
#include "memory/tensor.hpp"
#include "common/zendnnl_global.hpp"
#include "operators/matmul/matmul_context.hpp"
#include "operators/matmul/matmul_operator.hpp"
#include "operators/reorder/reorder_context.hpp"
#include "operators/reorder/reorder_operator.hpp"


#define MATMUL_SIZE_START 1
#define MATMUL_SIZE_END 1000

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;
using namespace zendnnl::ops;

using StorageParam = std::variant<std::pair<size_t, void *>, tensor_t>;

/** @brief Matmul Op Parameters Structure */
struct MatmulType {
  uint64_t matmul_m;
  uint64_t matmul_k;
  uint64_t matmul_n;
  uint32_t po_index;
  MatmulType();
};

extern int gtest_argc;
extern char **gtest_argv;
extern const uint32_t po_size; //Supported postop
extern int seed;
extern const uint32_t TEST_NUM;
extern const float MATMUL_F32_TOL;
extern const float MATMUL_BF16_TOL;
extern std::vector<MatmulType> matmul_test;

// TODO: Unify the tensor_factory in examples and gtest
//To generate random tensor
class tensor_factory_t {
 public:
  /** @brief Index type */
  using index_type = tensor_t::index_type;
  using data_type  = data_type_t;

  /** @brief zero tensor */
  tensor_t zero_tensor(const std::vector<index_type> size_, data_type dtype_);

  /** @brief uniformly distributed tensor */
  tensor_t uniform_dist_tensor(const std::vector<index_type> size_,
                               data_type dtype_,
                               float range_);

  /** @brief blocked tensor */
  tensor_t blocked_tensor(const std::vector<index_type> size_, data_type dtype_,
                          StorageParam param);
};

//Supported Postops declaration
extern std::vector<post_op_type_t> po_arr;
extern std::unordered_map<std::string, int> po_map;

/** @fn matmul_kernel_test
 *  @brief Compute Matmul Operation using AOCL kernel.
 *
 *  This function computes fused matmul that uses the Matmul Operator fused
 *  with randomly selected postop (supported by library) with AOCL kernel.
 *
 * */
void matmul_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                        tensor_t &bias, tensor_t &output_tensor, uint32_t index);

/** @fn matmul_forced_ref_kernel_test
 *  @brief Compute Matmul Op using Reference kernel.
 *
 *  This function computes fused matmul that uses the Matmul Operator fused
 *  with randomly selected postop (supported by library) with Reference kernel
 *  that only supports F32 datatype.
 *
 * */
void matmul_forced_ref_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                                   tensor_t &bias, tensor_t &output_tensor, uint32_t index);

/** @fn compare_tensor_2D
 *  @brief Function to compare two 2D tensor
 *
 * */
// ToDO: Replace with comparator operator
void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &flag);

/** @fn reorder_kernel_test
 *  @brief Function to Reorder tensor
 *
 *  This function reorders the tensor either by Inplace or OutofPlace.
 *  @return Reorderd tensor
 *
 * */
tensor_t reorder_kernel_test(tensor_t &input_tensor, bool inplace_reorder);
#endif
