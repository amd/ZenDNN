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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP

#include <cstdint>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Native BRGEMM over a native-layout weight: bf16 x bf16, u8 / s8 (or a
/// bf16 / f32 source quantized dynamically) x s8, and f32 x f32 to an f32 or
/// bf16 dst.  A call it turns away before any work returns unimplemented or
/// isa_unsupported (failure for bad arguments), with *why (when why is not
/// null) naming the reason, and leaves dst and the weight as they were;
/// once work started, every error is failure.  desc.K, desc.N and
/// weight.pack_nr must satisfy native_layout_fits (the layout rule's shape);
/// a release build does not check them.  With a GEMV choice (gemv.kernel
/// not none, decide_native_gemv's for the call) the call runs on that GEMV
/// kernel over weight.weight, its raw weight, instead (run_native_gemv); the
/// layout rule and pack_nr do not apply to it (an ldb below the weight row
/// length is turned away with unimplemented, as from the tiles), unless an
/// in-place rewrite of the buffer replaced the raw weight before the
/// kernel's pack read it: the call then runs on the native tiles from the
/// image, under the layout rule (failure when it does not take the call,
/// which no other route can run).
status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params,
        const native_gemv_plan_t &gemv = native_gemv_plan_t {},
        const char **why = nullptr);

/// The status native_brgemm_execute would turn the call away with (success
/// when it would run it), with *why naming the reason, without running it:
/// every check up to the acquisition of the weight's layout.  dst and the
/// weight are not touched; a dynamic quantization of a bf16 / f32 source
/// goes into the call's own buffers, and the scales and zero points it
/// makes into the buffers of params.
status_t native_brgemm_check(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params,
        const native_gemv_plan_t &gemv, const char **why);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP
