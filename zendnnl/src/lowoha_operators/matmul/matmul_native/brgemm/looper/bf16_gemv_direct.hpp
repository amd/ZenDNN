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

#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Direct BF16 GEMV (M=1) fast path.
/// Bypasses Planner + Looper for eligible shapes: single-thread uses a global
/// or thread-local packed B when it fits L2; multi-thread uses per-thread
/// column slices (OpenMP team size capped by \c m1_gemv_cap_threads) with
/// optional CCX-aware slice permutation on AMD Zen when \c nt % ccx_cores == 0.
/// Falls back to bf16_brgemm_execute when gates fail.
///
/// Returns true if the fast path handled the operation, false to fall back.
bool bf16_gemv_direct(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_BF16_GEMV_DIRECT_HPP
