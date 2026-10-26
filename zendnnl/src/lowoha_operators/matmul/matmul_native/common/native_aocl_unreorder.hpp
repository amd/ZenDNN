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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_AOCL_UNREORDER_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_AOCL_UNREORDER_HPP

#include <cstddef>
#include <memory>

#include "common/error_status.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// Frees memory from zendnnl_aligned_alloc.
struct native_aligned_free_t {
    void operator()(void *p) const { zendnnl_aligned_free(p); }
};

/// The aocl_dlp_blocked prepack of the call's weight has an inverse: a bf16
/// weight, or a static u8 x s8 one, prepacked without a DLP M hint.
bool aocl_prepack_unreorderable(const matmul_params &params);

/// Unreorders an aocl_dlp_blocked prepack of a K x N weight into raw, which
/// then holds the plain weight with rows ldb elements apart.
status_t aocl_prepack_unreorder(const void *aocl_packed, int K, int N, int ldb,
        bool transB, bool int8_u8s8, int num_threads,
        std::unique_ptr<void, native_aligned_free_t> &raw);

/// Bytes of the aocl_dlp_blocked prepack of a K x N weight.
size_t aocl_prepack_bytes(int K, int N, int ldb, bool transB, bool int8_u8s8);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_AOCL_UNREORDER_HPP
