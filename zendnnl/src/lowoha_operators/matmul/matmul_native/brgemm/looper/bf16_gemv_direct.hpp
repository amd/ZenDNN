/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_BF16_GEMV_DIRECT_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_BF16_GEMV_DIRECT_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The BF16 GEMV kernel of an M = 1 call of one thread (none for any other
/// call): a bf16 or f32 dst, on a CPU with AVX512-BF16, no transposed
/// source, at most one post-op that the kernels fuse (scan_gemv_postops),
/// no bias or a bf16 / f32 one, and ldc >= N; then
///   * N <= kBf16GemvNarrowMaxN and a [K, N] weight: bf16_narrow;
///   * else bf16_bkc, the pack cached globally at a weight cache mode other
///     than 0, else made for the call, unless the weight is not constant and
///     the pack exceeds the L2.
native_gemv_plan_t bf16_gemv_choice(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params);

/// Runs plan, bf16_gemv_choice's for desc, under the call's weight cache
/// mode: see run_native_gemv.
status_t bf16_gemv_run(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, int32_t weight_cache_mode);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_BF16_GEMV_DIRECT_HPP
