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

#include "lowoha_matmul.hpp"

namespace zendnnl {
namespace lowoha {

#if ZENDNNL_DEPENDS_AOCLDLP
/**
 * @brief Creates DLP (Deep Learning Post-op) metadata for post-operations.
 *
 * This function initializes and returns a pointer to `dlp_metadata_t` that
 * encapsulates the post-operation metadata for matrix multiplication.
 *
 * @param lowoha_po The parameters for the low-overhead matrix multiplication.
 * @param bias Pointer to the bias data.
 * @param dtypes Data types for the source, weight, and destination tensors.
 * @param N The number of columns in the output matrix.
 * @return Pointer to the created `dlp_metadata_t` object.
 */
dlp_metadata_t* create_dlp_post_op(const lowoha_params &lowoha_po, const void *bias, const data_types &dtypes, int N);

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
 * @param lowoha_po The parameters for the low-overhead matrix multiplication.
 * @param bias Pointer to the bias data.
 * @param dtypes Data types for the source, weight, and destination tensors.
 * @param N The number of columns in the output matrix.
 * @return Pointer to the created `aocl_post_op` object.
 */
aocl_post_op* create_blis_post_op(const lowoha_params &lowoha_po, const void *bias, const data_types &dtypes, int N);

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

} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_MATMUL_UTILS_HPP