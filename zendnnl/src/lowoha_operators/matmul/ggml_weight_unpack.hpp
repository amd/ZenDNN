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

#ifndef GGML_WEIGHT_UNPACK_HPP
#define GGML_WEIGHT_UNPACK_HPP

#include <cstdint>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Returns the minimum buffer size in bytes needed for
 *        ggml_unpack_weight_buffer.
 *
 * With use_bf16_scales=true the packed and unpacked totals are equal, so the
 * caller can pass the same allocation it already holds for the packed weights.
 * With use_bf16_scales=false (fp32 scales) the unpacked layout is larger and
 * a fresh allocation is required.
 *
 * @param ggml_type       GGML quantization type (8 = Q8_0, 2 = Q4_0)
 * @param use_bf16_scales true to use bf16 scales, false for fp32
 * @param M               number of rows in the weight matrix
 * @param K               number of columns (must be divisible by 32)
 * @return buffer size in bytes, or -1 on invalid parameters
 */
int64_t ggml_unpack_weight_buffer_size(int ggml_type, bool use_bf16_scales,
                                       int64_t M, int64_t K);

/**
 * @brief Unpack GGML quantised weights into a flat weight region + a flat
 *        scale region, both carved from a single caller-supplied buffer.
 *
 * On success:
 *   *wei_ptr  -> start of buf  (weight bytes)
 *   *scl_ptr  -> buf + weight region size  (scale bytes, fp32 or bf16)
 *
 * @return 0 on success, -1 on error.
 */
int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
                              bool is_superblock, bool use_bf16_scales,
                              bool use_unsigned_q4, int64_t M, int64_t K,
                              void *buf, int8_t **wei_ptr, void **scl_ptr);

/**
 * @brief Unpack GGML Q8_0 packed weights into separate weight and scale arrays.
 *
 * Uses an LRU cache keyed on (weight pointer, N, K) so that repeated calls
 * with the same packed buffer reuse the previously unpacked result.
 *
 * On success, @p weight is redirected to the flat int8 weight array and
 * params.quant_params.wei_scale is populated with the bf16 scale array.
 *
 * @param weight  [in/out] Pointer to packed weight data; updated to unpacked weights
 * @param N       Number of output channels (rows in the GGML weight matrix)
 * @param K       Number of input features (columns, must be divisible by 32)
 * @param params  [in/out] matmul_params whose wei_scale is populated on success
 * @return status_t::success on success, status_t::failure on error
 */
status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
                                      matmul_params &params);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // GGML_WEIGHT_UNPACK_HPP
