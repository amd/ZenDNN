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

#include "tensor_example.hpp"
#include "sample_example.hpp"
#include "matmul_example.hpp"
#include "batchmatmul_example.hpp"
#include "reorder_example.hpp"
#include "compare_op_example.hpp"
#include "embedding_bag_example.hpp"
#include "lowoha_matmul_example.hpp"


#define  OK          (0)
#define  NOT_OK      (1)

using namespace zendnnl::interface;
using namespace zendnnl::examples;

int main() {
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
  matmul_relu_forced_ref_kernel_example();
  matmul_broadcast_example(); //2d mm broadcast example
  matmul_mul_silu_mul_f32_kernel_example();
  matmul_silu_mul_bf16_kernel_example();
  matmul_strided_f32_kernel_example();
  run_lowoha_matmul_fp32_test();

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
   *  Demonstrates reordering memory from contiguous to blocked format,
   *  inplace reorder functionalities of Reorder operator.
   */
  reorder_outofplace_f32_kernel_example();
  reorder_outofplace_s8_kernel_example();
  reorder_outofplace_matmul_relu_f32_kernel_example();
  reorder_inplace_bf16_kernel_example();
  reorder_inplace_matmul_relu_bf16_kernel_example();

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

  /** Sample functionality examples.
   *
   */
  sample_f32_kernel_example();
  sample_bf16_kernel_example();

  return OK;
}
