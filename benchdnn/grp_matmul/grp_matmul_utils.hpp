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

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl

#endif
