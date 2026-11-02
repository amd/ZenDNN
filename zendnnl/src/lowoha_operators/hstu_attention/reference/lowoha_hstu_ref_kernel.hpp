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

#ifndef LOWOHA_HSTU_REF_KERNEL_HPP
#define LOWOHA_HSTU_REF_KERNEL_HPP

#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Scalar kernel, one (sequence, head, query row) per OpenMP iteration, FP32
/// accumulation, every mask entry evaluated. Supports every valid call.
status_t hstu_attention_reference(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params);

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_REF_KERNEL_HPP
