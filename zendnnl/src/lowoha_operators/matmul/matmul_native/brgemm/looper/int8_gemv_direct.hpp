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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_INT8_GEMV_DIRECT_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_INT8_GEMV_DIRECT_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The INT8 GEMV kernel of an M = 1 call (none for any other call): a u8 /
/// s8 source and an s8 weight to a bf16 / f32 dst on a CPU with
/// AVX512-VNNI, one thread, no transposed source, ldc == N, the BKC pack of
/// the weight (K in quads, N in BKC_NR_PAD columns) within the L2, at most
/// one post-op that the kernel fuses (scan_gemv_postops), a per-tensor f32
/// source scale and s32 / s8 / u8 zero point, f32 weight scales per tensor
/// or per column, no weight zero point, dst scale or dst zero point, and
/// alpha 1 and beta 0.
/// int8_bkc then: K-contiguous VNNI packing with vpdpbusd, the pack and its
/// column sums cached globally at a weight cache mode other than 0 (else
/// made for the call), and the dequantization per column (combined scale
/// and effective bias) computed for each call.
native_gemv_plan_t int8_gemv_choice(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params);

/// Runs plan, int8_gemv_choice's for desc and params, under the call's
/// weight cache mode: see run_native_gemv.
status_t int8_gemv_run(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, const matmul_params &params,
        int32_t weight_cache_mode);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_INT8_GEMV_DIRECT_HPP
