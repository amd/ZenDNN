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
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Native BRGEMM over a native-layout weight: bf16 x bf16, and u8 / s8
/// (or a bf16 / f32 source quantized dynamically) x s8.  A call it turns
/// away before any work returns unimplemented or isa_unsupported (the
/// caller may take another route unless the weight is an in-place image);
/// once work started, every error is failure.  desc.K, desc.N and
/// weight.pack_nr must satisfy native_layout_fits (the layout rule's shape);
/// a release build does not check them.
status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP
