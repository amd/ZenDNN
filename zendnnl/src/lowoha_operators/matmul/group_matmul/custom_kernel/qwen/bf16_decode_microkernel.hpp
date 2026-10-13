/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef ZENDNNL_GROUP_MATMUL_QWEN_BF16_DECODE_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_QWEN_BF16_DECODE_MICROKERNEL_HPP

#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/bf16_microkernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {
namespace decode_specialized {

ukernel_fn_t select_plain_bf16_one_panel(int MR);
ukernel_fn_t select_plain_bf16_fixed_k(int MR, int K);
ukernel_fn_t select_gated_bf16_one_panel(int MR, ActKind act);
ukernel_fn_t select_plain_bf16_f32_continue(int MR, int NV);

} // namespace decode_specialized
} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
