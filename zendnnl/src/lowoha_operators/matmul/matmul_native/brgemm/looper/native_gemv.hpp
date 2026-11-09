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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_GEMV_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_GEMV_HPP

#include <cstdint>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// The kernels native_brgemm_execute runs a call on: the native tiles over
/// the native layout, or a GEMV kernel on one thread over the call's raw
/// weight.
enum class native_gemv_t : uint8_t {
    none, ///< the native tiles
    bf16_narrow, ///< bf16_gemv_narrow, no pack
    bf16_bkc, ///< bf16_gemv_bkc over the BKC pack
    int8_bkc, ///< int8_gemv_bkc over the INT8 BKC pack
};

/// The name of kernel in the [MATMUL.NATIVE_BRGEMM] line.
const char *native_gemv_name(native_gemv_t kernel);

/// The GEMV choice of a call: the kernel and the post-op it fuses into its
/// store.
struct native_gemv_plan_t {
    native_gemv_t kernel = native_gemv_t::none;
    fused_postop_t fused = fused_postop_t::none;
};

/// The GEMV plan choice of an M = 1 call that asks for the native BRGEMM, on
/// its raw weight (none for any other call): for a call without a gated
/// activation, alpha 0 or a Swish of alpha other than 1, the kernel of
/// bf16_gemv_choice for a bf16 source and weight when the team of
/// m1_gemv_cap_threads over desc.num_threads is one thread, or of
/// int8_gemv_choice on one thread for a u8 / s8 source and an s8 weight;
/// then bf16_narrow, the only native kernel for N <= kBf16GemvNarrowMaxN,
/// and the BKC kernels for N a multiple of the native layout's blocks and a
/// constant weight within these bounds: bf16_bkc on a call of one thread,
/// or of more at K <= 1024; int8_bkc at N * K <= 64 Ki, or on a call of one
/// thread at K <= 1024.  A BKC kernel keeps its own pack of the weight at
/// weight cache mode 1 or 2, beside any native layout of the weight.  Reads
/// only desc, params and the CPU's cached parameters; no allocation.
native_gemv_plan_t decide_native_gemv(
        const GemmDescriptor &desc, const matmul_params &params);

/// Runs plan (kernel not none, decide_native_gemv's for desc and params) on
/// the raw weight on the calling thread, under the call's effective weight
/// cache mode (native_weight_ref_t::weight_cache_mode): at 1 or 2 the BKC
/// kernels read the pack cached for the weight pointer and the request,
/// packed again when the two ends of the weight's bytes no longer hold the
/// words the pack read (the bytes whose digests get_native_layout checks),
/// at 0 (and past ZENDNNL_LRU_CACHE_CAPACITY cached packs) a pack made for
/// this call.
/// Returns success; failure, with dst untouched, when a buffer cannot
/// be allocated before the kernel starts; unimplemented, with dst untouched,
/// when a bf16 kernel finds the buffer rewritten in place since the call
/// found its raw weight (the native tiles serve the call from the image).
status_t run_native_gemv(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, const matmul_params &params,
        int32_t weight_cache_mode);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_GEMV_HPP
