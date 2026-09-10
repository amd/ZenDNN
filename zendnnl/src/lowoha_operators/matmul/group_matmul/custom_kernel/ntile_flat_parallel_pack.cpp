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
 * @file ntile_flat_parallel_pack.cpp
 * @brief W8A8 grouped-MoE weight packing and scale widening.
 */

#include "ntile_flat_parallel_pack.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "common/zendnnl_compat.hpp"

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
#include <immintrin.h>
#include <omp.h>

#include "common/zendnnl_cpuid_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target( \
        "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512bf16,fma")
#elif defined(__clang__)
#pragma clang attribute push( \
        __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni," \
                              "avx512bf16,fma"))), \
        apply_to = function)
#endif

namespace {

// Pack one block of block_n output channels into the VNNI-ready layout:
//
//   [K/4][32][4] int8 quants
//   [32]         int32 compensation
//
// The compensation entry for channel n is 128 * sum_k w[n][k]: exactly the
// bias vpdpbusd introduces by treating the +128-shifted activation as
// unsigned.
inline void pack_weight_block(int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT dst,
        const int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT src, int64_t K) {
    const int64_t K4 = K / vnni_step;

    // k-major so the stores walk the destination linearly; the 32 source rows
    // stay resident as 32 cache lines.
    for (int64_t k4 = 0; k4 < K4; ++k4) {
        int8_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT d
                = dst + k4 * block_n * vnni_step;
        for (int64_t n = 0; n < block_n; ++n) {
            uint32_t v;
            std::memcpy(&v, src + n * K + k4 * vnni_step, sizeof(uint32_t));
            std::memcpy(d + n * vnni_step, &v, sizeof(v));
        }
    }

    // compensation: accumulate 128 * w over k with vpdpbusd against a constant
    // 0x80 unsigned operand, mirroring the runtime bias exactly.
    constexpr int cols = block_n / 16;
    alignas(64) __m512i vcomp[cols];
    for (int col = 0; col < cols; ++col) {
        vcomp[col] = _mm512_setzero_si512();
    }
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int64_t k4 = 0; k4 < K4; ++k4) {
        for (int col = 0; col < cols; ++col) {
            const __m512i vb
                    = _mm512_loadu_si512(reinterpret_cast<const void *>(
                            dst + k4 * block_n * vnni_step + col * 64));
            vcomp[col] = _mm512_dpbusd_epi32(vcomp[col], off, vb);
        }
    }
    for (int col = 0; col < cols; ++col) {
        _mm512_storeu_si512(
                reinterpret_cast<void *>(dst + block_n * K + col * 64),
                vcomp[col]);
    }
}

} // namespace

void widen_bf16_to_f32(float *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT out,
        const uint16_t *ZENDNNL_NTILE_FLAT_PARALLEL_RESTRICT in, int64_t size) {
    int64_t i = 0;
    for (; i + 16 <= size; i += 16) {
        const __m256i v
                = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
        _mm512_storeu_ps(out + i,
                _mm512_castsi512_ps(
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16)));
    }
    // Source scales are one value per routed row, so their count has no SIMD
    // alignment guarantee. Keep the weight-scale bulk loop above and handle
    // the arbitrary-M tail without reading or writing past either buffer.
    for (; i < size; ++i) {
        const uint32_t bits = static_cast<uint32_t>(in[i]) << 16;
        std::memcpy(out + i, &bits, sizeof(bits));
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma clang attribute pop
#endif

#endif // ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED

bool isa_supported() {
#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    const auto &uarch = native::detect_uarch();
    if (!uarch.avx512f || !uarch.avx512vnni || !uarch.avx512bf16) {
        return false;
    }
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    zendnnl_cpuid_count(7, 0, eax, ebx, ecx, edx);
    constexpr unsigned avx512dq = 1u << 17;
    constexpr unsigned avx512bw = 1u << 30;
    constexpr unsigned avx512vl = 1u << 31;
    return (ebx & (avx512dq | avx512bw | avx512vl))
            == (avx512dq | avx512bw | avx512vl);
#else
    return false;
#endif
}

status_t pack_weights(const int8_t *src, int8_t *dst, int64_t num_experts,
        int64_t out_channels, int64_t in_channels, int num_threads) {
    if (src == nullptr || dst == nullptr) { return status_t::op_bad_io; }
    if (num_experts <= 0 || out_channels <= 0 || in_channels <= 0) {
        return status_t::op_bad_io;
    }
    if (out_channels % block_n != 0 || in_channels % vnni_step != 0
            || in_channels > max_gemm_reduction) {
        return status_t::memory_bad_size;
    }

    const int64_t blocks_per_expert = out_channels / block_n;
    const int64_t oc_stride = packed_bytes_per_oc(in_channels);
    size_t total_blocks = 0;
    size_t packed_expert_bytes = 0;
    size_t raw_expert_bytes = 0;
    size_t packed_total_bytes = 0;
    size_t raw_total_bytes = 0;
    if (static_cast<uint64_t>(num_experts)
                    > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            || zendnnl_mul_overflow(static_cast<size_t>(num_experts),
                    static_cast<size_t>(blocks_per_expert), &total_blocks)
            || total_blocks
                    > static_cast<size_t>(std::numeric_limits<int64_t>::max())
            || zendnnl_mul_overflow(static_cast<size_t>(out_channels),
                    static_cast<size_t>(oc_stride), &packed_expert_bytes)
            || zendnnl_mul_overflow(static_cast<size_t>(out_channels),
                    static_cast<size_t>(in_channels), &raw_expert_bytes)
            || zendnnl_mul_overflow(static_cast<size_t>(num_experts),
                    packed_expert_bytes, &packed_total_bytes)
            || zendnnl_mul_overflow(static_cast<size_t>(num_experts),
                    raw_expert_bytes, &raw_total_bytes)) {
        return status_t::memory_bad_size;
    }
    if (!isa_supported()) { return status_t::isa_unsupported; }

#if ZENDNNL_NTILE_FLAT_PARALLEL_KERNELS_COMPILED
    const int64_t total = static_cast<int64_t>(total_blocks);
    const size_t packed_block_bytes
            = static_cast<size_t>(block_n) * static_cast<size_t>(oc_stride);
    const size_t raw_block_bytes
            = static_cast<size_t>(block_n) * static_cast<size_t>(in_channels);
    const int nth = num_threads > 0
            ? std::min(num_threads, omp_get_max_threads())
            : omp_get_max_threads();

#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t i = 0; i < total; ++i) {
        const size_t e = static_cast<size_t>(i / blocks_per_expert);
        const size_t nb = static_cast<size_t>(i % blocks_per_expert);
        pack_weight_block(
                dst + e * packed_expert_bytes + nb * packed_block_bytes,
                src + e * raw_expert_bytes + nb * raw_block_bytes, in_channels);
    }
    return status_t::success;
#else
    (void)num_threads;
    return status_t::isa_unsupported;
#endif
}

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
