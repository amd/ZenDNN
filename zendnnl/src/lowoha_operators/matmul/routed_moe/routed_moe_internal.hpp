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
 * @file routed_moe_internal.hpp
 * @brief Declarations shared between the routed-MoE translation units.
 *
 * Library-internal; not installed as part of the public API surface.
 */

#ifndef LOWOHA_ROUTED_MOE_INTERNAL_HPP
#define LOWOHA_ROUTED_MOE_INTERNAL_HPP

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "lowoha_operators/matmul/routed_moe/routed_moe.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace routed_moe {

// Keep the feature buildable on every platform supported by ZenDNN.  Native
// kernels are emitted only for x86-64 toolchains that provide AVX-512
// intrinsics; all other builds retain the public API and return
// status_t::isa_unsupported.
#ifndef ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
#if (defined(__x86_64__) || defined(_M_X64)) \
        && (defined(__GNUC__) || defined(__clang__))
#define ZENDNNL_ROUTED_MOE_KERNELS_COMPILED 1
#else
#define ZENDNNL_ROUTED_MOE_KERNELS_COMPILED 0
#endif
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ZENDNNL_ROUTED_RESTRICT __restrict
#else
#define ZENDNNL_ROUTED_RESTRICT __restrict__
#endif

// Blocking and packed-layout constants shared by validation and the native
// kernels.  Keeping these outside the intrinsic header lets the validator and
// non-x86 stubs compile without <immintrin.h>.
constexpr int64_t block_m = 32;
constexpr int64_t block_n = 32;
constexpr int64_t vnni_step = 4;
// Rows per micro-kernel pass.  Gate/up holds 2 streams x rows x 2 zmm
// accumulators plus 4 B vectors and the A broadcast (6 rows: 29 of 32 zmm);
// down holds rows x 2 accumulators, and 8 rows give enough independent chains
// to cover the vpdpbusd latency.
constexpr int64_t gate_up_kernel_rows = 6;
constexpr int64_t down_kernel_rows = 8;
constexpr int64_t max_kernel_rows = down_kernel_rows;
constexpr uint32_t packed_layout_version = 1;

// vpdpbusd accumulates an unsigned activation (at most 255) times a signed
// weight (magnitude at most 128) in int32.  This aligned ceiling prevents the
// raw accumulator from overflowing before compensation is subtracted.
constexpr int64_t max_gemm_reduction
        = (std::numeric_limits<int32_t>::max() / (255 * 128) / block_n)
        * block_n;
constexpr int64_t max_flat_routes = std::numeric_limits<int32_t>::max();
constexpr int64_t max_topk = max_flat_routes - (block_m - 1);
constexpr int64_t max_local_experts = (max_flat_routes - 1) / (block_m - 1);

/// The public name is retained for API compatibility.  This is the amortized
/// byte count per logical output channel, not a contiguous physical row.
constexpr int64_t packed_row_bytes_unchecked(int64_t in_channels) {
    return in_channels + static_cast<int64_t>(sizeof(int32_t));
}

struct checked_problem_sizes_t {
    int64_t numel = 0;
    int64_t gate_up_oc = 0;
    int64_t max_padded = 0;
    int64_t max_blocks = 0;
    size_t routing_elements = 0;
    size_t src_elements = 0;
    size_t intermediate_elements = 0;
    size_t down_elements = 0;
    size_t gate_up_packed_bytes = 0;
    size_t down_packed_bytes = 0;
};

/// Bit position helpers for the capability bitmasks.
constexpr uint32_t dt_bit(data_type_t dt) {
    return 1u << static_cast<uint32_t>(dt);
}
constexpr uint32_t act_bit(routed_moe_activation_t a) {
    return 1u << static_cast<uint32_t>(a);
}
constexpr uint32_t quant_bit(routed_moe_quant_t q) {
    return 1u << static_cast<uint32_t>(q);
}

/// Resolve the OMP width to run at.  Positive requests are clamped to the
/// runtime maximum so a malformed request cannot create an enormous team or
/// multiply scratch allocation sizes beyond the host's configured capacity.
int effective_num_threads(int32_t requested);

/// Runtime ISA predicate.  This includes OS AVX-512 state plus every
/// sub-feature emitted by the kernels (F/BW/DQ/VL, VNNI and BF16).
bool isa_supported();

/// Compute and range-check every geometry-derived count used by routing,
/// packing, pointer arithmetic and the main scratch buffers.
status_t checked_problem_sizes(
        const routed_moe_params &p, checked_problem_sizes_t &sizes);

/// Validate source/destination byte extents for the standalone pack API.
status_t checked_pack_sizes(int64_t num_experts, int64_t out_channels,
        int64_t in_channels, size_t &src_bytes, size_t &dst_bytes,
        int64_t &blocks_per_expert, int64_t &packed_bytes_per_oc);

/**
 * @brief Every validation rule that does not read caller data.
 *
 * Split out from the public validator so the executor can invoke exactly the
 * same checks before touching caller data or packed-cache state.
 */
status_t validate_static(const routed_moe_params &p);

/// Bound-check the routing ids (and the expert map entries they select).
status_t validate_routing_ids(const routed_moe_params &p);

/// Execute a validated routed-MoE problem.
status_t execute(const routed_moe_params &p);

/// Pack `[num_experts, out_channels, in_channels]` int8 weights into the
/// executor's block-VNNI layout.  Each block stores 32 output channels'
/// quants followed by their 32 int32 compensation values.
status_t pack_weights(const int8_t *src, int8_t *dst, int64_t num_experts,
        int64_t out_channels, int64_t in_channels, int64_t num_threads);

} // namespace routed_moe
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_ROUTED_MOE_INTERNAL_HPP
