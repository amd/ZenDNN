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

#ifndef _LOWOHA_REORDER_EXAMPLE_HPP_
#define _LOWOHA_REORDER_EXAMPLE_HPP_

#include "zendnnl.hpp"
#include <iostream>
#include <vector>

#define OK      (0)
#define NOT_OK  (1)

namespace zendnnl {
namespace examples {

using namespace zendnnl::lowoha::reorder;

//==============================================================================
// Basic Data Type Conversion Tests
//==============================================================================

/** @fn run_lowoha_reorder_bf16_to_int8_test
 *  @brief Demonstrates BF16 to INT8 quantization using LOWOHA reorder API.
 */
int run_lowoha_reorder_bf16_to_int8_test();

/** @fn run_lowoha_reorder_int8_to_bf16_test
 *  @brief Demonstrates INT8 to BF16 dequantization using LOWOHA reorder API.
 */
int run_lowoha_reorder_int8_to_bf16_test();

/** @fn run_lowoha_reorder_bf16_to_uint8_test
 *  @brief Demonstrates BF16 to UINT8 quantization using LOWOHA reorder API.
 */
int run_lowoha_reorder_bf16_to_uint8_test();

/** @fn run_lowoha_reorder_uint8_to_bf16_test
 *  @brief Demonstrates UINT8 to BF16 dequantization using LOWOHA reorder API.
 */
int run_lowoha_reorder_uint8_to_bf16_test();

//==============================================================================
// Granularity Tests
//==============================================================================

/** @fn run_lowoha_reorder_bf16_to_s8_per_tensor_test
 *  @brief Demonstrates BF16 to S8 per-tensor quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization with a single scale
 *  and zero-point for the entire tensor (per-tensor granularity).
 *
 *  Granularity: scale.dims = {}, zero_point.dims = {} (empty = per-tensor)
 */
int run_lowoha_reorder_bf16_to_s8_per_tensor_test();

/** @fn run_lowoha_reorder_bf16_to_s8_per_channel_test
 *  @brief Demonstrates BF16 to S8 per-channel quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization with different scale
 *  and zero-point per output channel (per-channel granularity).
 *
 *  Granularity: scale.dims = {num_channels}, zero_point.dims = {num_channels}
 */
int run_lowoha_reorder_bf16_to_s8_per_channel_test();

/** @fn run_lowoha_reorder_bf16_to_s8_per_group_test
 *  @brief Demonstrates BF16 to S8 per-group quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization with different scale
 *  and zero-point per group (per-group granularity).
 *
 *  Granularity: scale.dims = {num_groups, group_size}, zero_point.dims = {num_groups, group_size}
 */
int run_lowoha_reorder_bf16_to_s8_per_group_test();

/** @fn run_lowoha_reorder_bf16_to_s8_mixed_granularity_test
 *  @brief Demonstrates BF16 to S8 mixed granularity quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization with per-tensor scale
 *  and per-channel zero-point (mixed granularity).
 *
 *  Granularity: scale.dims = {} (per-tensor), zero_point.dims = {num_channels} (per-channel)
 */
int run_lowoha_reorder_bf16_to_s8_mixed_granularity_test();

/** @fn run_lowoha_reorder_bf16_to_s8_batched_test
 *  @brief Demonstrates BF16 to S8 batched matmul weight quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization for 3D batched tensor
 *  with a single per-tensor scale and zero-point shared across all batches.
 *
 *  Shape: [batch, M, N] with single scale/zp for all batches
 */
int run_lowoha_reorder_bf16_to_s8_batched_test();

//==============================================================================
// Dequantization Tests (INT8 -> BF16)
//==============================================================================

/** @fn run_lowoha_reorder_s8_to_bf16_per_tensor_test
 *  @brief Demonstrates S8 to BF16 per-tensor dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates INT8 to BF16 dequantization with a single scale
 *  and zero-point for the entire tensor (per-tensor granularity).
 */
int run_lowoha_reorder_s8_to_bf16_per_tensor_test();

/** @fn run_lowoha_reorder_s8_to_bf16_per_channel_test
 *  @brief Demonstrates S8 to BF16 per-channel dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates INT8 to BF16 dequantization with different scale
 *  and zero-point per output channel (per-channel granularity).
 */
int run_lowoha_reorder_s8_to_bf16_per_channel_test();

/** @fn run_lowoha_reorder_s8_to_bf16_per_group_test
 *  @brief Demonstrates S8 to BF16 per-group dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates INT8 to BF16 dequantization with different scale
 *  and zero-point per group (per-group granularity).
 */
int run_lowoha_reorder_s8_to_bf16_per_group_test();

/** @fn run_lowoha_reorder_s8_to_bf16_mixed_granularity_test
 *  @brief Demonstrates S8 to BF16 mixed granularity dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates INT8 to BF16 dequantization with per-tensor scale
 *  and per-channel zero-point (mixed granularity).
 */
int run_lowoha_reorder_s8_to_bf16_mixed_granularity_test();

/** @fn run_lowoha_reorder_s8_to_bf16_batched_test
 *  @brief Demonstrates S8 to BF16 batched dequantization using LOWOHA reorder API.
 *
 *  This example demonstrates INT8 to BF16 dequantization for 3D batched tensor
 *  with a single per-tensor scale and zero-point shared across all batches.
 */
int run_lowoha_reorder_s8_to_bf16_batched_test();

//==============================================================================
// Strided Memory Layout Tests
//==============================================================================

/** @fn run_lowoha_reorder_bf16_to_s8_strided_2d_test
 *  @brief Demonstrates BF16 to S8 strided 2D matrix quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization for a 2D matrix where
 *  the source data has non-contiguous memory layout (strided access).
 *
 *  Shape: [M, N] with custom strides [stride_M, stride_N]
 */
int run_lowoha_reorder_bf16_to_s8_strided_2d_test();

/** @fn run_lowoha_reorder_bf16_to_s8_strided_3d_test
 *  @brief Demonstrates BF16 to S8 strided 3D batched matrix quantization using LOWOHA reorder API.
 *
 *  This example demonstrates BF16 to INT8 quantization for a 3D batched matrix
 *  where the source data has non-contiguous memory layout (strided access).
 *
 *  Shape: [batch, M, N] with custom strides [stride_batch, stride_M, stride_N]
 */
int run_lowoha_reorder_bf16_to_s8_strided_3d_test();

/** @fn run_lowoha_reorder_bf16_to_s8_strided_row_padding_test
 *  @brief Demonstrates BF16 to S8 row-padded matrix quantization using LOWOHA reorder API.
 *
 *  This example shows a common real-world use case: a matrix with row padding
 *  for cache-line or SIMD alignment. The logical matrix is [M, N] but each row
 *  is padded to a larger stride for memory alignment.
 *
 *  Example: [4, 6] matrix padded to 8-element row stride for 64-byte alignment
 *  Shape: [M, N] = [4, 6], Strides: [8, 1] (each row padded to 8 elements)
 */
int run_lowoha_reorder_bf16_to_s8_strided_row_padding_test();

} // namespace examples
} // namespace zendnnl

#endif // _LOWOHA_REORDER_EXAMPLE_HPP_
