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
 * @brief Returns the minimum buffer size in bytes needed for the unpacked
 *        GGML weights + scales layout.
 *
 * With use_bf16_scales=true the packed and unpacked totals are equal, so the
 * inplace unpack in ggml_unpack_weight_buffer is safe to run directly on the
 * caller's packed-weights allocation. With use_bf16_scales=false (fp32 scales)
 * the unpacked layout is larger and a fresh allocation is required.
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
 * @brief Unpack GGML quantised weights inplace into a flat weight region + a
 *        flat scale region that share the original packed-weights allocation.
 *
 * The caller's @p weight_data buffer is reinterpreted as both the source of
 * packed blocks and the destination of the unpacked layout; temporary heap
 * buffers are used internally so that in-flight reads are not clobbered by
 * inplace writes. When use_bf16_scales=true, the packed and unpacked byte
 * counts match for Q8_0 / Q4_0, so the input buffer is always large enough.
 *
 * On success:
 *   *wei_ptr  -> start of the buffer (int8 weight bytes)
 *   *scl_ptr  -> weight region + weight_bytes (fp32 or bf16 scale bytes)
 *
 * @return 0 on success, -1 on error.
 */
int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
                              bool is_superblock, bool use_bf16_scales,
                              bool use_unsigned_q4, int64_t M, int64_t K,
                              int8_t **wei_ptr, void **scl_ptr);

/**
 * @brief Unpack GGML Q8_0 packed weights inplace into a flat int8 weight
 *        region followed by a bf16 scale region, reusing the caller's buffer.
 *
 * Uses an LRU cache keyed on (weight pointer, N, K) to avoid re-running the
 * transform if the same packed buffer is presented again. The cache value is
 * a sentinel bool (not a pointer) so LRU eviction never frees the caller's
 * weight memory.
 *
 * On success, the bytes at @p weight are transformed from packed GGML blocks
 * to [int8 weights | bf16 scales] (the pointer itself is unchanged because
 * the layout fits inplace for bf16 scales), and
 * params.quant_params.wei_scale is populated with the bf16 scale array that
 * lives immediately after the weight region.
 *
 * @param weight  [in/out] Pointer to packed weight data; contents are
 *                transformed inplace to the unpacked layout.
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
