/*******************************************************************************
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

#ifndef LOWOHA_MATMUL_UTILS_HPP
#define LOWOHA_MATMUL_UTILS_HPP

#include <utility>
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <class F>
inline void zendnnl_parallel_for(const int64_t begin, const int64_t end,
                                 const int64_t grain_size, const F &f) {

  if (begin >= end) {
    return;
  }
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  // choose number of tasks based on grain size and number of threads
  int64_t num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
  if (grain_size > 0) {
    num_threads = std::min(num_threads, divup((end - begin), grain_size));
  }

  #pragma omp parallel num_threads(num_threads)
  {
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      }
      catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}

inline const void *get_matrix_block(const void *base, int row_start,
                                    int col_start,
                                    int lda, bool trans, size_t type_size) {
  if (trans) {
    // Accessing column-major layout when transposed
    return static_cast<const uint8_t *>(base) + (col_start * lda + row_start) *
           type_size;
  }
  else {
    return static_cast<const uint8_t *>(base) + (row_start * lda + col_start) *
           type_size;
  }
}

inline void *get_output_block(void *base, int row_start, int col_start,
                              int ldc, size_t type_size) {
  return static_cast<uint8_t *>(base) + (row_start * ldc + col_start) * type_size;
}

inline int get_batch_index(int b, int batch_size) {
  return (batch_size == 1) ? 0 : (b % batch_size);
}

/**
* @brief Validates input parameters for matrix multiplication direct operation.
*
* This function performs comprehensive validation of all input parameters to ensure
* they are valid and compatible for matrix multiplication. It checks for null pointers,
* valid dimensions, supported data types, and parameter consistency.
*
* @param src Pointer to the source matrix A data
* @param weight Pointer to the weight matrix B data
* @param dst Pointer to the destination matrix C data
* @param M Number of rows in matrix A and output matrix C
* @param N Number of columns in matrix B and output matrix C
* @param K Number of columns in matrix A and rows in matrix B
* @param Batch_A Number of batches for matrix A
* @param Batch_B Number of batches for matrix B
* @param params Const reference to lowoha_params containing operation configuration
* @return status_t::success if all validations pass, status_t::failure otherwise
*/
status_t validate_matmul_direct_inputs(const void *src, const void *weight,
                                       const void *dst,
                                       const int M, const int N, const int K,
                                       const int Batch_A, const int Batch_B,
                                       const lowoha_params &params);

using get_reorder_buff_size_func_ptr = long unsigned int (*)(const char,
                                       const char, const char, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
  ,dlp_metadata_t *
#endif
                                                            );

template <typename T>
using reorder_func_ptr = void (*)(const char, const char, const char, const T *,
                                  T *, const md_t, const md_t, const md_t
#if ZENDNNL_DEPENDS_AOCLDLP
  ,dlp_metadata_t *
#endif
                                 );

/**
* @brief Reorders and caches weight matrices for optimized access patterns.
*
* This template function performs weight reordering to optimize memory access patterns
* for specific GEMM kernels and implements a caching mechanism to avoid redundant
* reordering operations for the same weight tensors.
*
* @tparam T Data type of the weights (float, int16_t, etc.)
* @param key Unique key identifying the weight tensor and reordering parameters
* @param weights Pointer to the original weight data
* @param reorder_weights Reference to pointer that will hold the reordered weights
* @param k Matrix dimension K (inner dimension)
* @param n Matrix dimension N (output dimension)
* @param ldb Leading dimension of matrix B
* @param order Memory layout order ('r' for row-major, 'c' for column-major)
* @param trans Transpose flag ('t' for transposed, 'n' for not transposed)
* @param mem_format_b Memory format for matrix B
* @param get_reorder_buf_size Function pointer to calculate required buffer size
* @param reorder_func Function pointer to perform the actual reordering
* @param weight_cache_type Type of caching strategy to use
* @return true if reordering was performed (cache miss), false if cached version was used
*/
template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
                            void *&reorder_weights, const int k, const int n, const int ldb,
                            const char order, const char trans, char mem_format_b,
                            get_reorder_buff_size_func_ptr get_reorder_buf_size,
                            reorder_func_ptr<T> reorder_func, int weight_cache_type);

/**
 * @brief Convert post-op names to a comma-separated string.
 *
 * This function takes a lowoha_params structure and converts all post-op types
 * to a comma-separated string representation.
 *
 * @param params The lowoha_params structure containing post-op information.
 * @return A string containing comma-separated post-op names, or "none" if no post-ops.
 */
std::string post_op_names_to_string(const lowoha_params &params);

/**
 * @brief Convert matmul_algo_t enum to string representation.
 *
 * This function converts a matmul_algo_t enum value to its string representation.
 *
 * @param kernel The matmul_algo_t enum value to convert.
 * @return A const char* pointer to the string representation of the kernel type.
 */
const char *kernel_to_string(matmul_algo_t kernel);

/**
 * @brief Convert data_type_t enum to string representation.
 *
 * This function converts a data_type_t enum value to its string representation.
 *
 * @param dtype The data_type_t enum value to convert.
 * @return A const char* pointer to the string representation of the data type.
 */
const char *data_type_to_string(data_type_t dtype);

/**
 * @brief Get post-op data types as a comma-separated string for binary_add/binary_mul.
 *
 * This function extracts data types from post-ops that are binary_add or binary_mul
 * and returns them as a comma-separated string.
 *
 * @param params The lowoha_params structure containing post-op information.
 * @return A string containing comma-separated data types, or empty string if none.
 */
std::string post_op_data_types_to_string(const lowoha_params &params);

inline bool may_i_use_blis_partition(int batch_count, int M, int N,
                                     int num_threads, data_type_t dtype);

inline matmul_algo_t select_algo_by_heuristics_bf16_bmm(int BS, int M, int N, int K,
    int num_threads);

inline matmul_algo_t select_algo_by_heuristics_bf16_mm(int M, int N, int K);

/**
* @brief Selects the optimal matrix multiplication kernel algorithm.
*
* This function analyzes the matrix multiplication parameters and system characteristics
* to select the most appropriate kernel algorithm for optimal performance. It considers
* factors such as matrix dimensions, batch sizes, data types, available hardware features,
* and library dependencies.
*
* @param params Reference to lowoha_params containing matrix multiplication configuration
* @param Batch_A Number of batches for matrix A (input tensor)
* @param Batch_B Number of batches for matrix B (weight tensor)
* @param batch_count Total number of batch operations to perform
* @param M Number of rows in matrix A and output matrix C
* @param N Number of columns in matrix B and output matrix C
* @param K Number of columns in matrix A and rows in matrix B
* @param num_threads Number of available threads for parallel execution
* @param bias Pointer to bias data; nullptr if no bias is used
* @param is_weights_const Indicates if the weights are constant
* @return matmul_algo_t The selected kernel algorithm (e.g., AOCL BLIS, OneDNN, LibXSMM)
*/
matmul_algo_t kernel_select(lowoha_params &params, int Batch_A, int Batch_B,
                            int batch_count, int M, int N, int K, int num_threads, const void *bias,
                            const bool is_weights_const);

} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_MATMUL_UTILS_HPP