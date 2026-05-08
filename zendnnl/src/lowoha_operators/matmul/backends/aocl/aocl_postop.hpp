/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZENDNNL_AOCL_POSTOP_HPP_
#define ZENDNNL_AOCL_POSTOP_HPP_

#include "lowoha_operators/matmul/lowoha_common.hpp"

#include "aocl_dlp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Shared scalar constants used as pointer-targets for AOCL/DLP post-op fields
inline constexpr float LEAKY_RELU_SLOPE_DEFAULT = 0.01f;
inline constexpr float ONE_F32            = 1.0f;

// Cast a (const) float address into the non-const void* slot expected by the
// AOCL DLP API. The kernels treat these scalars as read-only inputs, so the
// const_cast does not introduce undefined behavior and the underlying storage
// may safely live in read-only memory (e.g. inline constexpr globals).
inline void *get_void_ptr(const float &v) {
  return const_cast<void *>(static_cast<const void *>(&v));
}

// Helper function to compute number of elements from dimension vector.
// Returns 1 for empty dims (per-tensor case) specific for DLP use case,
// or product of all dims.
inline size_t get_num_elements(const std::vector<int64_t> &dims) {
  if (dims.empty()) {
    return 1;
  }
  size_t count = 1;
  for (auto d : dims) {
    count *= static_cast<size_t>(d);
  }
  return count;
}

/**
 * @brief Creates DLP metadata for post-operations
 *
 * This function initializes and returns a pointer to dlp_metadata_t that
 * encapsulates the post-operation metadata for matrix multiplication
 *
 * @param lowoha_param The parameters containing the post-operations chain specification
 * @param bias Pointer to the bias data to be added (can be nullptr if no bias)
 * @param dtypes Data types structure specifying source, weight, and destination tensor types
 * @param N The number of columns in the output matrix
 * @param K The number of columns in the input matrix / rows in weight matrix
 * @param M The number of rows in the output matrix (used for INT8 zero-point compensation)
 * @param zp_comp_acc Pointer to zero-point compensation buffer (nullptr if no ZP compensation)
 * @param zp_comp_ndim Dimensionality of ZP compensation: 0=none, 1=bias(N), 2=matrix(M*N)
 * @param kernel Algorithm selection for GEMM execution
 * @return Pointer to the created dlp_metadata_t object
 */
dlp_metadata_t *create_dlp_post_op(const matmul_params &lowoha_param,
                                   const void *bias, const matmul_data_types &dtypes, int N, int K,
                                   int M = 0, int32_t *zp_comp_acc = nullptr, int zp_comp_ndim = 0,
                                   zendnnl::ops::matmul_algo_t kernel = zendnnl::ops::matmul_algo_t::aocl_dlp);

/**
* @brief Cleans up DLP (Deep Learning Primitives) metadata.
*
* This function releases the resources allocated for the `dlp_metadata_t`
* object used in post-operations.
*
* @param aocl_po Pointer to the `dlp_metadata_t` object to be cleaned up.
*/
void cleanup_dlp_post_op(dlp_metadata_t *aocl_po);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif //ZENDNNL_AOCL_POSTOP_HPP_
