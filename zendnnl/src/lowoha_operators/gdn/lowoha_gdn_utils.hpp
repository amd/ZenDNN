/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef _LOWOHA_GDN_UTILS_HPP
#define _LOWOHA_GDN_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include "lowoha_operators/gdn/lowoha_gdn_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {

// Limits of the native kernels; larger shapes use the reference kernel.
constexpr int64_t kMaxHeadDim = 256;
constexpr int64_t kMaxConvWidth = 8;

// Tokens per chunk of the native chunked prefill. Fewer than FLA's 64: the
// products within a chunk cost per token in proportion to its length, the
// state products do not, and on CPU the saving outweighs the per-chunk work.
constexpr int64_t kChunkLen = 32;

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

inline float bf16_bits_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round to nearest even, NaN kept quiet.
inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t abs = bits & 0x7fffffffu;
    if (abs > 0x7f800000u) { return static_cast<uint16_t>(sign | 0x7e00u); }
    if (abs >= 0x477ff000u) { return static_cast<uint16_t>(sign | 0x7c00u); }
    if (abs < 0x38800000u) {
        // f16 subnormal: multiples of 2^-24, rounded to nearest even.
        const float scaled = std::ldexp(std::fabs(f), 24);
        const uint32_t half = static_cast<uint32_t>(std::nearbyint(scaled));
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t h = ((abs - 0x38000000u) >> 13);
    const uint32_t rem = abs & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) { ++h; }
    return static_cast<uint16_t>(sign | h);
}

inline float load_f32(const void *p, data_type_t dt, int64_t i) {
    switch (dt) {
        case data_type_t::f32: return static_cast<const float *>(p)[i];
        case data_type_t::bf16:
            return bf16_bits_to_f32(static_cast<const uint16_t *>(p)[i]);
        case data_type_t::f16:
            return f16_bits_to_f32(static_cast<const uint16_t *>(p)[i]);
        default: return 0.0f;
    }
}

inline void store_f32(void *p, data_type_t dt, int64_t i, float v) {
    switch (dt) {
        case data_type_t::f32: static_cast<float *>(p)[i] = v; break;
        case data_type_t::bf16:
            static_cast<uint16_t *>(p)[i] = f32_to_bf16_bits(v);
            break;
        case data_type_t::f16:
            static_cast<uint16_t *>(p)[i] = f32_to_f16_bits(v);
            break;
        default: break;
    }
}

// Value after a round trip through dt.
inline float round_to(float v, data_type_t dt) {
    switch (dt) {
        case data_type_t::bf16: return bf16_bits_to_f32(f32_to_bf16_bits(v));
        case data_type_t::f16: return f16_bits_to_f32(f32_to_f16_bits(v));
        default: return v;
    }
}

inline int64_t dt_size(data_type_t dt) {
    return dt == data_type_t::f32 ? 4 : 2;
}

inline bool is_gdn_float(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

inline float softplus_f32(float x, float threshold) {
    return x > threshold ? x : std::log1p(std::exp(x));
}

inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float norm_gate_f32(float z, gdn_norm_gate_t gate) {
    return gate == gdn_norm_gate_t::sigmoid ? sigmoid_f32(z) : silu_f32(z);
}

// ---------------------------------------------------------------------------
// Kernel entry points (internal)
// ---------------------------------------------------------------------------

/// The tensors of one gdn_direct call, with A_log and dt_bias in FP32.
struct gdn_tensors_t {
    const void *mixed_qkv;
    const void *a;
    const void *b;
    const float *A_log;
    const float *dt_bias;
    const void *conv_weight;
    const void *conv_bias;
    void *conv_state;
    void *ssm_state;
    const int32_t *state_indices;
    void *output;
};

/// Decode sequences [0, n) of the batch: sequence i owns token row i.
status_t gdn_decode_reference(
        const gdn_tensors_t &io, int64_t n, const gdn_params &params);
status_t gdn_decode_native(
        const gdn_tensors_t &io, int64_t n, const gdn_params &params);

/// Speculative verify of sequences [0, n) (params.spec_state_indices set):
/// sequence i owns token rows [offsets[i], offsets[i + 1]).
status_t gdn_verify_reference(const gdn_tensors_t &io, int64_t n,
        const int32_t *offsets, const gdn_params &params);
status_t gdn_verify_native(const gdn_tensors_t &io, int64_t n,
        const int32_t *offsets, const gdn_params &params);

/// Prefill sequences [s0, s1) of the batch: sequence i owns token rows
/// [offsets[i], offsets[i + 1]) and reads params.has_initial_state[i].
status_t gdn_prefill_reference(const gdn_tensors_t &io, int64_t s0, int64_t s1,
        const int32_t *offsets, const gdn_params &params);
/// Value blocks of the native chunked prefill. Its (sequence, value head)
/// pairs are numbered sequence by sequence, value head by value head; the
/// pairs from first_split on run in `blocks` blocks of head_v_dim / blocks
/// columns (a multiple of 32), the pairs before it as whole heads.
struct gdn_chunk_split_t {
    int64_t blocks;
    int64_t first_split;
};

/// algo is recurrent or chunked, as resolved by gdn_direct; split is read
/// by the chunked form only.
status_t gdn_prefill_native(const gdn_tensors_t &io, int64_t s0, int64_t s1,
        const int32_t *offsets, gdn_prefill_algo_t algo,
        const gdn_chunk_split_t &split, const gdn_params &params);

/// The split with the shortest estimated run time of the native chunked
/// prefill of sequences [s0, s1) on nthr threads (lowoha_gdn.cpp).
gdn_chunk_split_t gdn_chunk_split(const gdn_params &params,
        const int32_t *state_indices, const int32_t *offsets, int64_t s0,
        int64_t s1, int32_t nthr);

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_GDN_UTILS_HPP
