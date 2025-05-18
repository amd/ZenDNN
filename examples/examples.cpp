/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "compare_op_example.hpp"

#define  OK          (0)
#define  NOT_OK      (1)

using namespace zendnnl::examples;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;

int main() {
  tensor_unaligned_allocation_example();
  tensor_aligned_allocation_example();
  tensor_strided_aligned_allocation_example();
  tensor_copy_and_compare_example();
  tensor_move_and_refcount_example();
  tensor_constness_example();
  tensor_create_alike_example();
  sample_f32_kernel_example();
  sample_bf16_kernel_example();
  matmul_relu_f32_kernel_example();
  matmul_relu_bf16_kernel_example();
  matmul_relu_forced_ref_kernel_example();
  compare_op_example();
  compare_ref_and_aocl_matmul_kernel_example();

  return OK;
}
