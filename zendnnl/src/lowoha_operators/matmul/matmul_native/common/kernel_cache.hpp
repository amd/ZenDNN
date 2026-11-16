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

#ifndef MATMUL_NATIVE_KERNEL_CACHE_HPP
#define MATMUL_NATIVE_KERNEL_CACHE_HPP

#include <cstddef>
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The capacity (in layouts) of the native layout cache: matmul_config_t's
/// ZENDNNL_LRU_CACHE_CAPACITY, UINT32_MAX (unbounded) by default.  The cache
/// refuses a new layout, and the call packs one for itself, once the layouts
/// it holds, dropped ones included, reach it; it never evicts one.  Defined
/// in kernel_cache.cpp to avoid including common/op_config.hpp from this
/// header.
uint32_t get_weight_cache_capacity();

/// VNNI pair constant: 2 BF16 values per element position per k-pair.
inline constexpr int VNNI_PAIR = 2;

/// INT8 VNNI group size for vpdpbusd.
inline constexpr int INT8_VNNI_GRP = 4;

/// Clear all weight caches, and the kept buffers of widened F16 sources
/// (release_native_f16_sources). Must only be called when no thread is using
/// any previously returned pointer (e.g., between test cases or model swap).
/// Defined in kernel_cache.cpp.
void clear_all_weight_caches();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_KERNEL_CACHE_HPP
