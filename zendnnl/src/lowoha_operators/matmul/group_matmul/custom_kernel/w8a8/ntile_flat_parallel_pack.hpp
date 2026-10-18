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

/**
 * @file ntile_flat_parallel_pack.hpp
 * @brief Internal W8A8 grouped-MoE weight packing and layout utilities.
 *
 * The packed layout is consumed by the W8A8 MoE microkernels in the sibling
 * `ukernel` subtree. It remains library-internal and is not installed.
 */

#ifndef LOWOHA_CUSTOM_KERNEL_NTILE_FLAT_PARALLEL_PACK_HPP
#define LOWOHA_CUSTOM_KERNEL_NTILE_FLAT_PARALLEL_PACK_HPP

#include <cstdint>
#include <limits>

#include "common/error_status.hpp"

// Emit kernels only for x86-64 toolchains that provide AVX-512 intrinsics.
// Every other build keeps the decline-and-fall-through behaviour, so the
// library still compiles and behaves identically to before this feature.
#ifndef ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
#if (defined(__x86_64__) || defined(_M_X64)) \
        && (defined(__GNUC__) || defined(__clang__))
#define ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED 1
#else
#define ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED 0
#endif
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT __restrict
#else
#define ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT __restrict__
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

using zendnnl::error_handling::status_t;

// block_n = 32 is the width the epilogues are written for: two 16-lane f32
// vectors combine into one 32-lane bf16 store, and the compensation / scale
// loads are two full zmm each.
constexpr int64_t block_n = 32;
constexpr int64_t vnni_step = 4;

/// vpdpbusd accumulates an unsigned activation (at most 255) times a signed
/// weight (magnitude at most 128) into int32. This aligned ceiling keeps the
/// raw accumulator from overflowing before compensation is subtracted.
constexpr int64_t max_gemm_reduction
        = (std::numeric_limits<int32_t>::max() / (255 * 128) / block_n)
        * block_n;

/// Amortized packed bytes per logical output channel: the channel's `K`
/// quants plus its int32 compensation. Physical storage is block-major, so
/// this is a stride/size quantity rather than a contiguous row length.
constexpr int64_t packed_bytes_per_oc(int64_t in_channels) {
    return in_channels + static_cast<int64_t>(sizeof(int32_t));
}

/// Runtime ISA predicate: AVX-512 F/BW/DQ/VL plus VNNI and BF16.
bool isa_supported();

/**
 * @brief Pack `[num_experts, out_channels, in_channels]` int8 weights into
 *        the executor's block-VNNI layout.
 *
 * For each block of @c block_n output channels the layout is
 *
 *     [K/4][32][4] int8 quants        (32 * K bytes)
 *     [32]         int32 compensation (128 bytes)
 *
 * Exposed so the unit tests can assert the layout against an independent
 * scalar packer.
 */
status_t pack_weights(const int8_t *src, int8_t *dst, int64_t num_experts,
        int64_t out_channels, int64_t in_channels, int num_threads);

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
/// Widen an arbitrary-length bf16 scale vector to f32.
void widen_bf16_to_f32(float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT out,
        const uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT in, int64_t size);
#endif

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_CUSTOM_KERNEL_NTILE_FLAT_PARALLEL_PACK_HPP
