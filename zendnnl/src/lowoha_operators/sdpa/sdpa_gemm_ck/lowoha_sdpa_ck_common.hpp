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

/// Data-type independent pieces of the SDPA custom-kernel paths: tile
/// geometry, scratch growth and call counting. Internal to sdpa_gemm_ck.

#ifndef LOWOHA_SDPA_CK_COMMON_HPP
#define LOWOHA_SDPA_CK_COMMON_HPP

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
namespace ck {

/// Packed B columns per block: two zmm of FP32 accumulators per row.
inline constexpr int kNR = 32;
/// Rows per tile: kMR x 2 accumulators.
inline constexpr int kMR = 8;

/// Grow a per-thread buffer to at least `need` elements. False when the
/// allocation fails, so the caller declines to AOCL-DLP instead of throwing
/// out of its OpenMP region.
template <typename T>
bool grow(std::vector<T> &v, size_t need) noexcept {
    if (v.size() >= need) return true;
    try {
        v.resize(need);
    } catch (const std::exception &) { return false; }
    return true;
}

/// Count a BF16 / INT8 GEMM call taken by CK while `test_api::count_calls`
/// is on.
void count_bf16_call();
void count_int8_call();

} // namespace ck
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif
