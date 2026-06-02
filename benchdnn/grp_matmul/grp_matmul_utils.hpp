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

#ifndef GRP_MATMUL_UTILS_HPP
#define GRP_MATMUL_UTILS_HPP

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include "common/data_types.hpp"

namespace zendnnl {
namespace benchdnn {
namespace grp_matmul {

using zendnnl::common::data_type_t;
using zendnnl::common::size_of;

// ── 64-byte aligned buffer ──────────────────────────────────────────────

struct AlignedBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    void alloc(size_t n) {
        free();
        bytes = (n + 63) & ~size_t(63);
        ptr = std::aligned_alloc(64, bytes);
    }
    void free() { if (ptr) { std::free(ptr); ptr = nullptr; } }
    ~AlignedBuffer() { free(); }
    AlignedBuffer() = default;
    AlignedBuffer(AlignedBuffer &&o) noexcept : ptr(o.ptr), bytes(o.bytes)
        { o.ptr = nullptr; }
    AlignedBuffer &operator=(AlignedBuffer &&o) noexcept
        { free(); ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; return *this; }
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
};

/// Configuration for one group matmul benchmark shape.
struct GrpMatmulConfig {
    int num_ops = 8;
    std::vector<int> M_per_op;   ///< per-expert M (size == num_ops)
    int K = 4096;
    int N = 14336;
    int iters = 200;
    data_type_t src_dt = data_type_t::bf16;
    data_type_t wei_dt = data_type_t::bf16;
    data_type_t dst_dt = data_type_t::bf16;
    bool is_weights_const = true;
    int warmup = 50;
    int moe_topk = 0;  ///< 0 = no MoE post-op, >0 = enable with this topk
    int gated_act = 0;  ///< 0 = off, 1 = silu_and_mul, 2 = gelu_and_mul, 3 = swiglu_oai_mul
    int N_down = 0;     ///< 0 = no fused down_proj, >0 = fused Op1→Act→Op2 with this N_down
    int use_internal_alloc = 0;  ///< 0 = caller-allocated dst/dst_down (legacy);
                                 ///< 1 = library-allocated Op1 scratch + src-reuse
                                 ///<     for Op2 output (requires N_down > 0).
    int total_experts = 0;       ///< Optional trailing field.  When > num_ops,
                                 ///< exercises the framework prepack-extras
                                 ///< contract: the dispatcher receives weight
                                 ///< buffers for all `total_experts` slots
                                 ///< (the first `num_ops` correspond to firing
                                 ///< experts; the rest are extras the prepack
                                 ///< module pre-warms).  When 0 / absent,
                                 ///< treated as `total_experts = num_ops` and
                                 ///< `params[i].active_matmul == total_matmul`
                                 ///< — legacy-equivalent behaviour.

    /// DQ-INT8 (dynamic-quant) toggle.  When non-zero the driver
    /// drives the N-tile / custom-kernel int8 path instead of the
    /// bf16 path:
    ///   * `params[i].dynamic_quant = true`
    ///   * `params[i].dtypes.compute = s8` (sym) or `u8` (asym),
    ///     selected by `compute_dt` below.
    ///   * `params[i].quant_params.wei_scale` populated with a per-
    ///     expert per-channel f32 buffer of length `N`.
    ///   * `params[i].quant_params.src_scale.buff = nullptr` —
    ///     the library's pre-OMP hoist (`HoistedSrcQuant` in
    ///     `n_tile/group_matmul_n_tile.cpp`) allocates and fills
    ///     the per-token scale at runtime, so the caller leaves it
    ///     null (matches the production DQ-INT8 contract; see
    ///     gtests/group_matmul/test_algos.cpp::TestGroupMatmulAuto
    ///     SelectAlgo_DynamicQuant).
    ///
    /// Requires:
    ///   * src_dt=bf16, wei_dt=s8, dst_dt=bf16
    ///     (CK's `resolve_variant()` truth table — see
    ///     custom_kernel/dispatch.cpp:resolve_variant).
    ///   * K % 4 == 0 (the int8 microkernel reduces along the K-
    ///     axis in 4-byte VPDPBUSD lanes; mirrored by the
    ///     `sym_k = (k/4)*4` clamp in test_quant.cpp).
    /// Refused at parse time when these preconditions are violated.
    int dynamic_quant = 0;       ///< 0 = bf16 path (default), 1 = DQ-INT8

    /// Compute dtype for the DQ-INT8 family (ignored when
    /// `dynamic_quant == 0`).  Drives `params[i].dtypes.compute`:
    ///   * data_type_t::s8 (default) — symmetric kernel
    ///     (`kS8_S8_BF16_SYM`), no src_zp produced by the hoist.
    ///   * data_type_t::u8           — asymmetric kernel
    ///     (`kU8_S8_BF16_ASYM`); the hoist additionally allocates
    ///     and fills a per-token src_zp.
    /// Stored as the underlying `data_type_t` enum so the driver
    /// passes it straight through; the parser accepts the strings
    /// "s8" / "u8" for human-friendly input files.
    data_type_t compute_dt = data_type_t::s8;

    int max_M() const { return *std::max_element(M_per_op.begin(), M_per_op.end()); }
    int total_M() const { return std::accumulate(M_per_op.begin(), M_per_op.end(), 0); }
    bool is_uniform_M() const {
        return std::all_of(M_per_op.begin(), M_per_op.end(),
                           [&](int m){ return m == M_per_op[0]; });
    }
};

/// Parse M field: single int or colon-separated per-expert list.
std::vector<int> parse_M(const std::string &s, int num_ops);

/// Parse one CSV line into a GrpMatmulConfig. Returns true on success.
bool parse_config(const std::string &line, GrpMatmulConfig &cfg);

/// Format M for display: uniform → "4", heterogeneous → "15-323(128)".
std::string format_M(const GrpMatmulConfig &cfg);

/// Fill buffer with valid BF16 random values in [-1.0, 1.0].
void fill_bf16_random(void *buf, size_t elems, uint32_t seed);

/// Fill buffer with random data appropriate for the given data type.
void fill_buffer(void *buf, size_t elems, data_type_t dt, uint32_t seed);

/// Fill an f32 buffer with positive scales clustered around 1/127 —
/// matches the magnitude a real per-token / per-channel quant scale
/// produced by `max(|x|) / 127` would have for bf16 inputs in
/// [-1, 1].  Used by the benchdnn DQ-INT8 path to populate
/// `params[i].quant_params.wei_scale.buff`; the per-token src_scale
/// is hoist-allocated by the library so the driver does not fill it.
void fill_quant_scale_f32(float *buf, size_t elems, uint32_t seed);

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl

#endif
