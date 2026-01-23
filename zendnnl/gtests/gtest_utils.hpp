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
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

#define MATMUL_SIZE_START 1
#define MATMUL_SIZE_END 3000
#define MATMUL_LARGE_SIZE_END 10000
#define BATCH_START 1
#define BATCH_END 256
#define TEST_PARTITIONS 3
#define ENABLE_F32_RELAXATION 0

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;
using namespace zendnnl::ops;
using namespace zendnnl::lowoha::matmul;
using namespace zendnnl::lowoha::reorder;
using namespace zendnnl::lowoha::embag;

using StorageParam = std::variant<std::pair<size_t, void *>, tensor_t>;

/** @brief Matmul Op Parameters Structure */
struct MatmulType {
  uint64_t matmul_m;
  uint64_t matmul_k;
  uint64_t matmul_n;
  post_op_type_t po_type;
  bool     transA;
  bool     transB;
  //TODO: Add support for other data_types as well
  float    alpha;
  float    beta;
  bool     use_LOWOHA;
  matmul_algo_t algo = matmul_algo_t::none;
  data_type_t source_dtype;
  data_type_t output_dtype;
  quant_granularity_t weight_granularity;
  uint32_t num_threads;
  MatmulType(uint32_t test_index = 0, uint32_t total_tests = 1);
};

/** @brief BatchMatmul Op Parameters Structure */
struct BatchMatmulType {
  uint64_t batch_size;
  MatmulType mat{};
  BatchMatmulType(uint32_t test_index = 0, uint32_t total_tests = 1);
};

struct ReorderType {
  bool inplace_reorder;
  MatmulType mat{};
  ReorderType(uint32_t test_index = 0, uint32_t total_tests = 1);
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
  data_type_t indices_dtype;
  data_type_t offsets_dtype;
  bool fp16_scale_bias;
  bool strided;
  bool use_LOWOHA;
  uint32_t num_threads;
  EmbagType();
};

/** @brief Embedding Op Parameters Structure */
struct EmbeddingType {
  uint64_t num_embeddings;
  uint64_t embedding_dim;
  uint64_t num_indices;
  int64_t padding_index;
  bool is_weights;
  data_type_t indices_dtype;
  bool fp16_scale_bias;
  bool strided;
  bool use_LOWOHA;
  uint32_t num_threads;
  EmbeddingType();
};

extern int gtest_argc;
extern char **gtest_argv;
extern const uint32_t po_size; //Supported postop
extern const uint32_t dtype_size;
extern int64_t seed;
extern std::string cmd_post_op;
extern std::string cmd_backend;
extern std::string cmd_lowoha;
extern uint32_t cmd_num_threads;
extern const float MATMUL_F32_TOL;
extern const float MATMUL_BF16_TOL;
extern const float REORDER_TOL;
extern const float EMBAG_F32_TOL;
extern const float EMBAG_BF16_TOL;
extern const float EMBAG_INT4_TOL;
extern const float epsilon_f32;
extern const float epsilon_bf16;
extern const float rtol_f32;
extern const float rtol_bf16;
extern const float epsilon_woq;
extern const float rtol_woq;
extern std::vector<MatmulType> matmul_test;
extern std::vector<BatchMatmulType> batchmatmul_test;
extern std::vector<ReorderType> reorder_test;
extern std::vector<EmbagType> embag_test;
extern std::vector<EmbeddingType> embedding_test;

// TODO: Unify the tensor_factory in examples and gtest
//To generate random tensor
class tensor_factory_t {
 public:
  /** @brief Index type */
  using index_type = tensor_t::index_type;
  using data_type  = data_type_t;

  /** @brief zero tensor */
  tensor_t zero_tensor(const std::vector<index_type> size_, data_type dtype_,
                       tensor_t scale = tensor_t(), tensor_t zp = tensor_t(),
                       bool strided = false);

  /** @brief uniformly distributed tensor */

  tensor_t uniform_dist_tensor(const std::vector<index_type> size_,
                               data_type dtype_,
                               float val, bool trans = false,
                               tensor_t scale = tensor_t(), tensor_t zp = tensor_t());

  /** @brief uniformly distributed strided tensor */
  tensor_t uniform_dist_strided_tensor(const std::vector<index_type> size_,
                                       const std::vector<index_type> stride_,
                                       data_type dtype_, float range_, bool trans = false,
                                       tensor_t scale = tensor_t(), tensor_t zp = tensor_t());

  /** @brief uniform tensor with optional transpose and quantization support */
  tensor_t uniform_tensor(const std::vector<index_type> size_, data_type dtype_,
                          float val_, std::string tensor_name_="uniform",
                          bool trans = false,
                          tensor_t scale = tensor_t(), tensor_t zp = tensor_t());

  /** @brief blocked tensor */
  tensor_t blocked_tensor(const std::vector<index_type> size_, data_type dtype_,
                          float val);

  /** @brief copy tensor */
  tensor_t copy_tensor(const std::vector<index_type> size_, data_type dtype_,
                       StorageParam param, bool trans, bool is_blocked);

  /** @brief Generate random indices tensor with optional padding index */
  tensor_t random_indices_tensor(const std::vector<index_type> size_,
                                 uint64_t num_embeddings_, data_type_t indices_dtype_);

  /** @brief Generate random offsets tensor for bag boundaries */
  tensor_t random_offsets_tensor(const std::vector<index_type> size_,
                                 uint64_t num_indices_, data_type_t offsets_dtype_,
                                 bool include_last_offset_ = true);

  /** @brief Generate quantized random table tensor for embedding & embag */
  tensor_t quantized_embedding_tensor_random(const std::vector<index_type> size_,
      data_type dtype_, std::string tensor_name_="quant random",
      bool fp16_scale_bias = true, float scale_min = 0.10,
      float scale_max = 0.19, float bias_min = 0, float bias_max = 7);
};

/**
 * @class Parser
 * @brief Command Line Parser Utility for gtest
 *
 * This class provides functionality to read command line arguments and properly
 * handles them.
 *
 */
class Parser {
  /** @brief Map to store command line arguments as {key,val} pair */
  std::unordered_map<std::string, std::string> umap {};
  /** @brief check if string is numeric or not */
  bool isInteger(const std::string &s);
  /** @brief read from key if valid or invalid key is given */
  void read_from_umap(const std::string &key, int64_t &num);
  void read_from_umap(const std::string &key, uint32_t &num);
  void read_from_umap(const std::string &key, std::string &num);
 public:
  /** @brief to make object callable */
  void operator()(const int &argc,
                  char *argv[],
                  int64_t &seed, uint32_t &test_num, std::string &po, std::string &backend,
                  std::string &ai_test_mode,
                  std::string &lowoha, uint32_t &num_threads, std::string &input_file,
                  std::string &op, uint32_t &ndims);
};

bool is_binary_postop(post_op_type_t post_op);
// Array of supported post operations
const post_op_type_t post_op_arr[] = {
  post_op_type_t::relu,
  post_op_type_t::gelu_tanh,
  post_op_type_t::gelu_erf,
  post_op_type_t::sigmoid,
  post_op_type_t::swish,
  post_op_type_t::tanh,
  post_op_type_t::binary_add,
  post_op_type_t::binary_mul,
  post_op_type_t::none
};

//Supported Dtype declaration
extern std::vector<data_type_t> dtype_arr;

/** @fn strToAlgo
 *  @brief Convert string representation to matmul algorithm type
 *
 *  This function converts a string representation of a matrix multiplication
 *  algorithm to the corresponding matmul_algo_t enumeration value.
 *
 *  @param str String representation of the algorithm (e.g., "onednn", "aocl_dlp", etc.)
 *  @return matmul_algo_t The corresponding algorithm enumeration value
 */
matmul_algo_t strToAlgo(std::string str);

/** @fn algoToStr
 *  @brief Convert matmul algorithm type to string representation
 *
 *  This function converts a matmul_algo_t enumeration value to its
 *  corresponding string representation for display or logging purposes.
 *
 *  @param algo The matmul algorithm enumeration value
 *  @return std::string String representation of the algorithm
 */
std::string algoToStr(matmul_algo_t algo);

/**
* @fn strToPostOps
* @brief Converts a string representation of a post operation to its corresponding enum value
*
* This function translates a string such as "relu", "gelu_tanh" into the
* corresponding `post_op_type_t`
*
* @param str String representation of the post operation (e.g., "relu", "gelu_tanh").
* @return post_op_type_t Corresponding enum value.
*/
post_op_type_t strToPostOps(const std::string &str);

/**
 * @brief Converts a post_op_type_t enum value to its string representation.
 *
 * This function maps a post_op_type_t value (e.g., relu, gelu_tanh) to its corresponding
 * string (e.g., "relu", "gelu_tanh") for display or output purposes.
 *
 * @param post_op The post_op_type_t enum value to convert.
 * @return std::string The string representation of the post operation.
 */
std::string postOpsToStr(post_op_type_t post_op);
/** @fn read_matmul_inputs
 *  @brief Read and parse matmul test configurations from a file
 *
 *  This function reads a file containing matmul test parameters and returns
 *  a vector of BatchMatmulType configurations. It handles both 2D matmul (ndims=2)
 *  and 3D batch matmul (ndims=3) test cases from the same function.
 *
 *  For 2D matmul (ndims=2), input format: M,K,N,postOp,kernel,transA,transB,alpha,beta (9 fields)
 *  For 3D matmul (ndims=3), input format: BS,M,K,N,postOp,kernel,transA,transB,alpha,beta (10 fields)
 *
 *  @param file Path to the input file
 *  @param ndims Dimension of matmul operation (2 for matmul, 3 for batch matmul). Default is 2.
 *  @return std::vector<BatchMatmulType> Vector of parsed matmul configurations
 */
std::vector<BatchMatmulType> read_matmul_inputs(const std::string &file,
    uint32_t ndims = 2);

/** @fn read_reorder_inputs
 *  @brief Read and parse reorder test configurations from a file
 *
 *  This function reads a file containing reorder test parameters and returns
 *  a vector of ReorderType configurations. Reorder operations transform tensor
 *  layouts for optimized memory access patterns.
 *
 *  Input format: M,K,N,postOp,kernel,transA,transB,inplace_reorder (8 fields)
 *
 *  @param file Path to the input file
 *  @return std::vector<ReorderType> Vector of parsed reorder configurations
 */
std::vector<ReorderType> read_reorder_inputs(const std::string &file);

/** @fn trim
 *  @brief Remove leading and trailing whitespace from a string
 *
 *  This helper function removes all leading and trailing whitespace characters
 *  from the input string. The string is modified in-place.
 *
 *  @param str Reference to the string to be trimmed (modified in-place)
 */
void trim(std::string &str);

/** @fn split
 *  @brief Split a string into tokens based on a delimiter
 *
 *  This helper function splits a string into multiple substrings using the
 *  specified delimiter character. Empty tokens are preserved in the result.
 *
 *  @param s The input string to split
 *  @param delimiter The character to use as delimiter (typically ',')
 *  @return std::vector<std::string> Vector of tokenized substrings
 */
std::vector<std::string> split(const std::string &s, char delimiter);

/** @fn matmul_kernel_test
 *  @brief Compute Matmul Operation using AOCL kernel.
 *
 *  This function computes fused matmul that uses the Matmul Operator fused
 *  with randomly selected postop (supported by library) with AOCL kernel.
 *
 *  @return matmul status
 * */
status_t matmul_kernel_test(tensor_t &input_tensor, tensor_t &weights,
                            tensor_t &bias, tensor_t &output_tensor, post_op_type_t po_type,
                            tensor_t &binary_tensor, bool use_LOWOHA, matmul_algo_t algo,
                            float alpha = 1.0f,
                            float beta = 0.0f);

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
                                       post_op_type_t po_type, tensor_t &binary_tensor, bool use_LOWOHA,
                                       matmul_algo_t algo,
                                       float alpha = 1.0f,
                                       float beta = 0.0f);

/** @fn reorder_kernel_test
 *  @brief Function to Reorder tensor
 *
 *  This function reorders/unreorder the tensor either by Inplace or OutofPlace.
 *  @return Updated tensor and status
 *
 * */
std::pair<tensor_t, status_t> reorder_kernel_test(tensor_t &input_tensor,
    bool inplace_reorder, void **reorder_weights,
    data_type_t source_dtype = data_type_t::f32);

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
                           bool fp16_scale_bias,
                           bool use_LOWOHA=false);

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
                                      bool fp16_scale_bias);

/** @fn embedding_kernel_test
 *  @brief Test function for embedding kernel
 *
 * @return status_t Success or failure status
 */
status_t embedding_kernel_test(tensor_t &table_tensor,
                               tensor_t &indices_tensor,
                               tensor_t &weights_tensor,
                               tensor_t &output_tensor,
                               int64_t padding_index,
                               bool is_weights,
                               bool fp16_scale_bias,
                               bool use_LOWOHA=false);

/** @fn embedding_forced_ref_kernel_test
 *  @brief Test function for embedding reference kernel (forced)
 *
 * @return status_t Success or failure status
 */
status_t embedding_forced_ref_kernel_test(tensor_t &table_tensor,
    tensor_t &indices_tensor,
    tensor_t &weights_tensor,
    tensor_t &output_tensor,
    int64_t padding_index,
    bool is_weights,
    bool fp16_scale_bias);

/** @fn compare_tensor_2D
 *  @brief Function to compare two 2D tensor
 *
 *  This function compares two 2D tensors element by element and checks if they are
 *  within a specified tolerance (Either Absolute or relative).
 *  @return void
 * */
// ToDO: Replace with comparator operator
void compare_tensor_2D(tensor_t &output_tensor, tensor_t &output_tensor_ref,
                       uint64_t m,
                       uint64_t n, const float tol, bool &is_comparison_successful);

/** @fn compare_tensor_2D_matrix
 *  @brief Function to compare two matrix result after matrix matmul
 *
 *  This function compares two 2D tensors representing matrix matmul element by element
 *  and checks if they are within a specified bound or not.
 *  Bound is based on error-propagation thoery.
 *  @return void
 *
 * */
void compare_tensor_2D_matrix(tensor_t &output_tensor,
                              tensor_t &output_tensor_ref, uint64_t m,
                              uint64_t n, uint64_t k, const float rtol,
                              const float epsilon, bool &flag,
                              bool enable_f32_relaxation = false,
                              bool is_woq = false);

/** @fn compare_tensor_3D_matrix
 *  @brief Function to compare two matrix result after batch-matrix matmul
 *
 *  This function compares two 3D tensors representing batch-matrix matmul
 *  element by element and checks if they are within a specified bound or not.
 *  Bound is based on error-propagation thoery.
 *  @return void
 *
 * */
void compare_tensor_3D_matrix(tensor_t &output_tensor,
                              tensor_t &output_tensor_ref, uint64_t batch_size,
                              uint64_t m, uint64_t n, uint64_t k, const float rtol,
                              const float epsilon, bool &flag,
                              bool enable_f32_relaxation = false);

/** @fn get_aligned_size
 *  @brief Function to align the given size_ according to the alignment
 *
 * */
size_t get_aligned_size(size_t alignment, size_t size_);
#endif
