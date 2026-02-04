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

#include "tensor_example.hpp"
#include "sample_example.hpp"
#include "matmul_example.hpp"
#include "batchmatmul_example.hpp"
#include "reorder_example.hpp"
#include "compare_op_example.hpp"
#include "embedding_bag_example.hpp"
#include "lowoha_matmul_example.hpp"
#include "lowoha_conv2d_example.hpp"
#include "lowoha_reorder_example.hpp"
#include "lowoha_softmax_example.hpp"
#include "lowoha_pooling_example.hpp"
#include <iostream>
#include "sdpa_example.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

using namespace zendnnl::interface;
using namespace zendnnl::examples;

int main() {
  try {
    /** Tensor functionality examples.
     *  Demonstrates strided, unaligned allocation, aligned allocation,
     *  constness of tensor functionalities of tensor.
     */
    tensor_unaligned_allocation_example();
    tensor_aligned_allocation_example();
    tensor_strided_aligned_allocation_example();
    tensor_copy_and_compare_example();
    tensor_move_and_refcount_example();
    tensor_constness_example();
    tensor_create_alike_example();
    tensor_broadcast_example();
    tensor_axes_permutation_example();
    tensor_quantization_example();

    /** MatMul operator functionality examples.
    *  Demonstrates fused post-ops, different data types computation,
    *  strided input MatMul functionalities of MatMul operator.
    */
    matmul_relu_f32_kernel_example();
    matmul_relu_bf16_kernel_example();
    matmul_silu_add_int8_kernel_example();
    matmul_mul_silu_mul_f32_kernel_example();
    matmul_silu_mul_bf16_kernel_example();
    matmul_strided_f32_kernel_example();
    matmul_relu_forced_ref_kernel_example();
    matmul_broadcast_example(); //2d mm broadcast example
    run_lowoha_matmul_fp32_test();
    matmul_woq_bf16_kernel_example();
    run_lowoha_matmul_woq_bf16s4_test();
    run_lowoha_matmul_int8_caching_test();

    /** LOWOHA Conv2D operator functionality examples.
     *  Demonstrates 2D convolution with low-overhead API including:
     *  - Basic FP32 and BF16 convolutions
     *  - Depthwise convolution (MobileNet pattern)
     *  - Strided convolution for downsampling
     *  - Dilated convolution (atrous)
     *  - Reference kernel implementation
     *  - OneDNN vs Reference accuracy comparison
     */
    run_lowoha_conv2d_fp32_test();
    run_lowoha_conv2d_bf16_test();
    run_lowoha_depthwise_conv2d_test();
    run_lowoha_strided_conv2d_test();
    run_lowoha_dilated_conv2d_test();

    /** LOWOHA Reorder operator functionality examples.
     *  Demonstrates data type conversion between BF16 and INT8/UINT8 using
     *  the low-overhead LOWOHA reorder API.
     */
    // BF16 <-> INT8/UINT8 tests
    run_lowoha_reorder_bf16_to_int8_test();
    run_lowoha_reorder_int8_to_bf16_test();
    run_lowoha_reorder_bf16_to_uint8_test();
    run_lowoha_reorder_uint8_to_bf16_test();
    run_lowoha_reorder_bf16_to_s8_per_tensor_test();
    run_lowoha_reorder_bf16_to_s8_per_channel_test();
    run_lowoha_reorder_bf16_to_s8_per_channel_row_test();
    run_lowoha_reorder_bf16_to_s8_per_group_test();
    run_lowoha_reorder_bf16_to_s8_per_group_col_test();
    run_lowoha_reorder_bf16_to_s8_mixed_granularity_test();
    run_lowoha_reorder_bf16_to_s8_mixed_row_group_test();
    run_lowoha_reorder_bf16_to_s8_batched_test();
    run_lowoha_reorder_s8_to_bf16_per_tensor_test();
    run_lowoha_reorder_s8_to_bf16_per_channel_test();
    run_lowoha_reorder_s8_to_bf16_per_channel_row_test();
    run_lowoha_reorder_s8_to_bf16_per_group_test();
    run_lowoha_reorder_s8_to_bf16_per_group_col_test();
    run_lowoha_reorder_s8_to_bf16_mixed_granularity_test();
    run_lowoha_reorder_s8_to_bf16_mixed_row_group_test();
    run_lowoha_reorder_bf16_to_s8_strided_2d_test();
    run_lowoha_reorder_bf16_to_s8_strided_3d_test();
    run_lowoha_reorder_bf16_to_s8_strided_row_padding_test();

    // FP32 <-> INT8/UINT8 tests
    run_lowoha_reorder_f32_to_int8_test();
    run_lowoha_reorder_int8_to_f32_test();
    run_lowoha_reorder_f32_to_uint8_test();
    run_lowoha_reorder_uint8_to_f32_test();
    run_lowoha_reorder_f32_to_s8_per_tensor_test();
    run_lowoha_reorder_f32_to_s8_per_channel_test();
    run_lowoha_reorder_f32_to_s8_per_channel_row_test();
    run_lowoha_reorder_f32_to_s8_per_group_test();
    run_lowoha_reorder_f32_to_s8_per_group_col_test();
    run_lowoha_reorder_f32_to_s8_mixed_granularity_test();
    run_lowoha_reorder_f32_to_s8_mixed_row_group_test();
    run_lowoha_reorder_f32_to_s8_batched_test();
    run_lowoha_reorder_s8_to_f32_per_tensor_test();
    run_lowoha_reorder_s8_to_f32_per_channel_test();
    run_lowoha_reorder_s8_to_f32_per_channel_row_test();
    run_lowoha_reorder_s8_to_f32_per_group_test();
    run_lowoha_reorder_s8_to_f32_per_group_col_test();
    run_lowoha_reorder_s8_to_f32_mixed_granularity_test();
    run_lowoha_reorder_s8_to_f32_mixed_row_group_test();
    run_lowoha_reorder_f32_to_s8_strided_2d_test();
    run_lowoha_reorder_f32_to_s8_strided_3d_test();
    run_lowoha_reorder_f32_to_s8_strided_row_padding_test();

    // FP32 <-> BF16 tests (with optional scale/zero-point)
    run_lowoha_reorder_f32_to_bf16_simple_test();
    run_lowoha_reorder_f32_to_bf16_with_scale_test();
    run_lowoha_reorder_bf16_to_f32_simple_test();
    run_lowoha_reorder_bf16_to_f32_with_scale_test();
    run_lowoha_reorder_f32_to_bf16_per_channel_test();
    run_lowoha_reorder_f32_to_bf16_per_group_test();
    run_lowoha_reorder_bf16_to_f32_per_channel_test();
    run_lowoha_reorder_f32_to_bf16_strided_2d_test();
    run_lowoha_reorder_f32_to_bf16_batched_test();

    /** LOWOHA Softmax operator functionality examples.
     *  Demonstrates softmax and log-softmax operations using the low-overhead
     *  LOWOHA API with support for multi-dimensional tensors.
     */
    run_lowoha_softmax_fp32_test();
    run_lowoha_softmax_bf16_test();

    /** LOWOHA Pooling operator functionality examples.
     *  Demonstrates max pooling and average pooling operations using the
     *  low-overhead LOWOHA API with various configurations.
     */
    run_lowoha_maxpool_fp32_test();
    run_lowoha_avgpool_fp32_test();
    run_lowoha_maxpool_bf16_test();
    run_lowoha_avgpool_padding_modes_test();

    /** BatchMatMul operator functionality examples.
     *  Demonstrates fused post-ops, different data types computation,
     */
    batch_matmul_relu_f32_kernel_example();
    batch_matmul_wei2d_relu_f32_kernel_example();
    batch_matmul_inp2d_relu_f32_kernel_example();
    batch_matmul_relu_bf16_kernel_example();
    batch_matmul_relu_forced_ref_kernel_example();
    batch_matmul_mul_silu_mul_f32_kernel_example();
    batch_matmul_silu_mul_bf16_kernel_example();
    batchmatmul_broadcast_example();

    /** Reorder operator functionality examples.
     *  Demonstrates reordering memory from contiguous to blocked format and vice versa,
     *  inplace reorder functionalities of Reorder operator.
     */
    reorder_outofplace_f32_kernel_contiguous_blocked_example();
    reorder_outofplace_s8_kernel_contiguous_blocked_example();
    reorder_outofplace_matmul_relu_f32_kernel_contiguous_blocked_example();
    reorder_inplace_bf16_kernel_contiguous_blocked_example();
    reorder_inplace_matmul_relu_bf16_kernel_contiguous_blocked_example();
    reorder_outofplace_bf16_kernel_blocked_contiguous_example();
    reorder_inplace_s8_kernel_blocked_contiguous_example();
    reorder_unreorder_outofplace_bf16_kernel_example();

    /** Compare operator functionality examples.
     *  Demonstrates compare operator usage for comparison of tensors.
     */
    compare_op_example();
    compare_ref_and_aocl_matmul_kernel_example();

    /** Embedding Bag operator functionality examples.
     *  Demonstrates embedding bag operator usage for efficient lookup of embeddings.
     */
    embedding_bag_f32_kernel_example();
    embedding_bag_f32_forced_ref_kernel_example();
    embedding_f32_kernel_example();
    embedding_bag_u4_kernel_example();
    embedding_bag_u4_ref_kernel_example();
    group_embedding_bag_direct_example();

    /** Sample functionality examples.
     *
     */
    sample_f32_kernel_example();
    sample_bf16_kernel_example();

    /** SDPA encoder functionality examples.
     *  Demonstrates SDPA encoder operator usage for transformer attention.
     */
    sdpa_example();

    return OK;
  }
  catch (const zendnnl::error_handling::exception_t &e) {
    std::cerr << "ZenDNN exception caught in main: " << e.what() << std::endl;
    return NOT_OK;
  }
  catch (const std::exception &e) {
    std::cerr << "Standard exception caught in main: " << e.what() << std::endl;
    return NOT_OK;
  }
  catch (...) {
    std::cerr << "Unknown exception caught in main" << std::endl;
    return NOT_OK;
  }
}
