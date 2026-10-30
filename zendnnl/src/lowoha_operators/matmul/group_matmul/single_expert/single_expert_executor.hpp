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

/// Single-expert EXECUTOR: runs a `SingleExpertBlockPlan` for a lone expert's
/// decode on the custom kernel, from a call view the N-tile executor fills.
/// Built only on `custom_kernel/dispatch.hpp`'s tile and block pieces, so a
/// new blocking scheme changes this file and the plan, not the shared
/// executors.  Library-internal.

#ifndef ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_EXECUTOR_HPP
#define ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_EXECUTOR_HPP

#include <cstdint>

#include "../custom_kernel/dispatch.hpp"
#include "single_expert_plan.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// One custom-kernel call of a lone expert.  For DQ-INT8 `src` is the
/// quantized source and the scale and epilogue fields are the N-tile hoist's;
/// BF16 leaves them null.  `dst` is the caller's wide [M, N] output.
struct SingleExpertCallView {
    const custom_kernel::CallContext *kctx = nullptr;
    int M = 0;
    int N = 0;
    int K = 0;
    const void *src = nullptr;
    int lda = 0;
    const void *src_scale = nullptr;
    const int32_t *src_zp = nullptr;
    const void *wei_scale = nullptr;
    const custom_kernel::Int8EpilogueConfig *int8_epilogue = nullptr;
    const void *bias = nullptr;
    void *dst = nullptr;
    int ldc = 0;
    int num_threads = 0;
};

/// Whether the executor can run this call (the block pieces support it).
bool single_expert_call_supported(const SingleExpertCallView &call);

/// Run `plan` on `call`.  False when the OpenMP team came up short; the
/// destination is then incomplete and the caller must fail the call.
bool single_expert_plan_execute(
        const SingleExpertCallView &call, const SingleExpertBlockPlan &plan);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_EXECUTOR_HPP
