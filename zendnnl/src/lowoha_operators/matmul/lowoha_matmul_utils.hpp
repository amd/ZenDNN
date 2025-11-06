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
#include "lowoha_matmul.hpp"

#if ZENDNNL_DEPENDS_ONEDNN
  #include "operators/matmul/onednn/matmul_onednn_kernel.hpp"
  using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {

#if ZENDNNL_DEPENDS_AOCLDLP
  /**
  * @brief Creates DLP (Deep Learning Post-op) metadata for post-operations.
  *
  * This function initializes and returns a pointer to `dlp_metadata_t` that
  * encapsulates the post-operation metadata for matrix multiplication.
  *
  * @param lowoha_param The parameters for the low-overhead matrix multiplication.
  * @param bias Pointer to the bias data.
  * @param dtypes Data types for the source, weight, and destination tensors.
  * @param N The number of columns in the output matrix.
  * @return Pointer to the created `dlp_metadata_t` object.
  */
  dlp_metadata_t *create_dlp_post_op(const lowoha_params &lowoha_param,
  const void *bias, const data_types &dtypes, int N);

  /**
  * @brief Cleans up DLP (Deep Learning Post-op) metadata.
  *
  * This function releases the resources allocated for the `dlp_metadata_t`
  * object used in post-operations.
  *
  * @param aocl_po Pointer to the `dlp_metadata_t` object to be cleaned up.
  * @param post_op The parameters for the post-operation.
  */
  void cleanup_dlp_post_op(dlp_metadata_t *aocl_po, const lowoha_params &post_op);

#else
  /**
  * @brief Creates BLIS (Basic Linear Algebra Subprograms) post-op metadata.
  *
  * This function initializes and returns a pointer to `aocl_post_op` that
  * encapsulates the post-operation metadata for matrix multiplication.
  *
  * @param lowoha_param The parameters for the low-overhead matrix multiplication.
  * @param bias Pointer to the bias data.
  * @param dtypes Data types for the source, weight, and destination tensors.
  * @param N The number of columns in the output matrix.
  * @return Pointer to the created `aocl_post_op` object.
  */
  aocl_post_op *create_blis_post_op(const lowoha_params &lowoha_param,
  const void *bias, const data_types &dtypes, int N);

  /**
  * @brief Cleans up BLIS (Basic Linear Algebra Subprograms) post-op metadata.
  *
  * This function releases the resources allocated for the `aocl_post_op`
  * object used in post-operations.
  *
  * @param aocl_po Pointer to the `aocl_post_op` object to be cleaned up.
  * @param post_op The parameters for the post-operation.
  */
  void cleanup_blis_post_op(aocl_post_op *aocl_po, const lowoha_params &post_op);
#endif

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

#if ZENDNNL_DEPENDS_ONEDNN
bool reorderAndCacheWeights(Key_matmul key,
                             onednn_utils_t::onednn_matmul_params &dnnl_params, int weight_cache_type,
                             dnnl::engine &eng);

void reorderWeights(onednn_utils_t::onednn_matmul_params &dnnl_params,
                    dnnl::engine &eng);
#endif

} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_MATMUL_UTILS_HPP