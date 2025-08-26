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
#include "operators/embag/embag_context.hpp"
#include "operators/embag/embag_operator.hpp"


#define MATMUL_SIZE_START 1
#define MATMUL_SIZE_END 1000
#define BATCH_START 1
#define BATCH_END 256

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
  bool     transA;
  bool     transB;
  //TODO: Add support for other data_types as well
  float    alpha;
  float    beta;
  MatmulType();
};

/** @brief BatchMatmul Op Parameters Structure */
struct BatchMatmulType {
  uint64_t batch_size;
  MatmulType mat{};
  BatchMatmulType();
};

/** @brief Embag Op Parameters Structure */
struct EmbagType {
  uint64_t num_embeddings;
  uint64_t embedding_dim;
  uint64_t num_bags;
  uint64_t num_indices;
  embag_algo_t algo;
  int64_t padding_index;
  bool include_last_offset;
  bool is_weights;
  int64_t scatter_stride;
  EmbagType();
};

extern int gtest_argc;
extern char **gtest_argv;
extern const uint32_t po_size; //Supported postop
extern int seed;
extern const uint32_t TEST_NUM;
extern const float MATMUL_F32_TOL;
extern const float MATMUL_BF16_TOL;
extern const float EMBAG_F32_TOL;
extern const float EMBAG_BF16_TOL;
extern std::vector<MatmulType> matmul_test;
extern std::vector<BatchMatmulType> batchmatmul_test;
extern std::vector<EmbagType> embag_test;

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
                               float range_, bool trans = false);

  /** @brief uniformly distributed strided tensor */
  tensor_t uniform_dist_strided_tensor(const std::vector<index_type> size_,
                                       const std::vector<index_type> stride_,
                                       data_type dtype_, float range_, bool trans);

  /** @brief blocked tensor */
  tensor_t blocked_tensor(const std::vector<index_type> size_, data_type dtype_,
                          StorageParam param);

  /** @brief Generate random indices tensor with optional padding index */
  tensor_t random_indices_tensor(const std::vector<index_type> size_,
                                 uint64_t num_embeddings);

  /** @brief Generate random offsets tensor for bag boundaries */
  tensor_t random_offsets_tensor(const std::vector<index_type> size_,
                                 uint64_t num_indices, bool include_last_offset = true);
};

bool is_binary_postop(const std::string post_op);

//Supported Postops declaration
extern std::vector<std::pair<std::string, post_op_type_t>> po_arr;

/** @fn matmul_kernel_test
 *  @brief Compute Matmul Operation using AOCL kernel.
 *
 *  This function computes fused matmul that uses the Matmul Operator fused
 *  with randomly selected postop (supported by library) with AOCL kernel.
 *
 *  @return matmul status
 * */
status_t matmul_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                            tensor_t &bias, tensor_t &output_tensor, uint32_t index,
                            tensor_t &binary_tensor, float alpha = 1.0f, float beta = 0.0f);

/** @fn matmul_forced_ref_kernel_test
 *  @brief Compute Matmul Op using Reference kernel.
 *
 *  This function computes fused matmul that uses the Matmul Operator fused
 *  with randomly selected postop (supported by library) with Reference kernel
 *  that only supports F32 datatype.
 *
 *  @return matmul status
 * */
status_t matmul_forced_ref_kernel_test(tensor_t &input_tensor,
                                       tensor_t &weights,
                                       tensor_t &bias, tensor_t &output_tensor,
                                       uint32_t index, tensor_t &binary_tensor, float alpha = 1.0f, float beta = 0.0f);

/** @fn reorder_kernel_test
 *  @brief Function to Reorder tensor
 *
 *  This function reorders the tensor either by Inplace or OutofPlace.
 *  @return Reordered tensor and reorder status
 *
 * */
std::pair<tensor_t, status_t> reorder_kernel_test(tensor_t &input_tensor,
    bool inplace_reorder, void **reorder_weights);

/** @fn embag_kernel_test
 *  @brief Test function for embag kernel
 *
 * @return status_t Success or failure status
 */
status_t embag_kernel_test(tensor_t &table_tensor,
                           tensor_t &indices_tensor,
                           tensor_t &offsets_tensor,
                           tensor_t &weights_tensor,
                           tensor_t &output_tensor,
                           embag_algo_t algo,
                           int64_t padding_index,
                           bool include_last_offset,
                           bool is_weights,
                           int64_t scatter_stride);

/** @fn embag_forced_ref_kernel_test
 *  @brief Test function for embag reference kernel (forced)
 *
 * @return status_t Success or failure status
 */
status_t embag_forced_ref_kernel_test(tensor_t &table_tensor,
                                      tensor_t &indices_tensor,
                                      tensor_t &offsets_tensor,
                                      tensor_t &weights_tensor,
                                      tensor_t &output_tensor,
                                      embag_algo_t algo,
                                      int64_t padding_index,
                                      bool include_last_offset,
                                      bool is_weights,
                                      int64_t scatter_stride);

/** @fn compare_tensor_2D
 *  @brief Function to compare two 2D tensor
 *
 * */
// ToDO: Replace with comparator operator
void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &flag);

/** @fn compare_tensor_3D
 *  @brief Function to compare two 3D tensor
 *
 * */
// ToDO: Replace with comparator operator
void compare_tensor_3D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t batch_size, uint64_t m, uint64_t n,
                       const float tol, bool &flag);

/** @fn get_aligned_size
 *  @brief Function to align the given size_ according to the alignment
 *
 * */
size_t get_aligned_size(size_t alignment, size_t size_);
#endif
