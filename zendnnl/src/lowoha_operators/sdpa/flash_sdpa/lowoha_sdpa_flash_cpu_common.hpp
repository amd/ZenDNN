/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

// Helpers shared by the FP (lowoha_sdpa_flash_cpu.cpp) and dynamic-INT8
// (lowoha_sdpa_flash_cpu_int8.cpp) flash-attention translation units.
// Everything here is a template or `inline`, so both TUs may include it.

#ifndef ZENDNNL_LOWOHA_SDPA_FLASH_CPU_COMMON_HPP
#define ZENDNNL_LOWOHA_SDPA_FLASH_CPU_COMMON_HPP

#include "lowoha_sdpa_flash_cpu.hpp"

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "lowoha_operators/common/simd_ops.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <type_traits>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

// Canonical reduced-precision numeric types from the common library.
// These are wrapper types rather than plain `uint16_t` aliases, so code in
// this file should follow the access conventions used by the common types and
// the helper routines below when reading or writing raw 16-bit payloads,
// instead of assuming generic interchangeability in unrelated contexts.
using bfloat16_t = zendnnl::common::bfloat16_t;
using float16_t = zendnnl::common::float16_t;

template <typename T>
inline void ensure_scratch_size(std::vector<T> &buffer, size_t elements) {
    if (buffer.size() < elements) { buffer.resize(elements); }
}

#define SDPA_SA_CHECK(cond, msg) \
    do { \
        if (!(cond)) throw std::invalid_argument(msg); \
    } while (0)

template <typename T>
struct is_reduced_fp : std::false_type {};
template <>
struct is_reduced_fp<bfloat16_t> : std::true_type {};
template <>
struct is_reduced_fp<float16_t> : std::true_type {};
template <typename T>
inline constexpr bool is_reduced_fp_v = is_reduced_fp<T>::value;

using accum_t = float;

template <typename T2>
inline float mask_elem_at(T2 const *b, int i) {
    if constexpr (std::is_same_v<T2, float>) {
        return b[i];
    } else {
        return static_cast<float>(b[i]);
    }
}

template <typename scalar_t>
inline scalar_t float_to_scalar(float f) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return f;
    } else {
        return scalar_t(f);
    }
}

inline void sdpa_data_index_init(int64_t linear_idx, int64_t &i, int64_t dim_i,
        int64_t &j, int64_t dim_j, int64_t &k, int64_t dim_k) {
    const int64_t slice = dim_j * dim_k;
    i = linear_idx / slice;
    int64_t rem = linear_idx % slice;
    j = rem / dim_k;
    k = rem % dim_k;
}

// All supported Tin variants (f32 / bf16 / f16) write an F32 output through
// the corresponding AOCL kernel (f32f32f32of32 / bf16bf16f32of32 /
// f16f16f32of32), so the output buffer is always `float *`.
template <typename Tin>
inline void zendnn_gemm(int64_t m, int64_t n, int64_t k, float alpha,
        const Tin *a, int64_t lda, const Tin *b, int64_t ldb, float beta,
        float *c, int64_t ldc, bool TransA, bool TransB) {
    zendnnl::lowoha::matmul::matmul_params params;
    zendnnl::lowoha::matmul::matmul_data_types matmul_dtype;
    matmul_dtype.bias = zendnnl::common::data_type_t::none;
    matmul_dtype.compute = zendnnl::common::data_type_t::none;
    matmul_dtype.dst = zendnnl::common::data_type_t::f32;
    if constexpr (std::is_same_v<Tin, float>) {
        matmul_dtype.src = zendnnl::common::data_type_t::f32;
        matmul_dtype.wei = zendnnl::common::data_type_t::f32;
    } else if constexpr (std::is_same_v<Tin, bfloat16_t>) {
        matmul_dtype.src = zendnnl::common::data_type_t::bf16;
        matmul_dtype.wei = zendnnl::common::data_type_t::bf16;
    } else {
        static_assert(std::is_same_v<Tin, float16_t>,
                "zendnn_gemm: only float / bfloat16_t / float16_t input "
                "supported");
        matmul_dtype.src = zendnnl::common::data_type_t::f16;
        matmul_dtype.wei = zendnnl::common::data_type_t::f16;
    }
    params.dtypes = matmul_dtype;
    params.lowoha_algo = zendnnl::ops::matmul_algo_t::aocl_dlp;

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    zendnnl::lowoha::matmul::matmul_direct('r', TransA, TransB, m, n, k, alpha,
            a, lda, b, ldb, nullptr, beta, c, ldc, false, batch_params, params);
}

// ---------------------------------------------------------------------------
// SIMD-templated helper functions.
// All functions below are parameterised on SimdTag (avx512_tag | scalar_tag).
//
// #pragma GCC target enables AVX-512 so that __m512 locals inside the
// avx512_tag instantiation use the correct ABI (zmm registers, 64-byte
// alignment).  The scalar_tag instantiation never touches __m512 types
// (VecF32 = struct{float}), so the target is harmless there.
//
// The target string intentionally omits "f16c": the only FP16 conversions
// used here are _mm512_cvtph_ps / _mm512_cvtps_ph, which live in
// <avx512fintrin.h> and require avx512f only. F16C covers the older
// 128/256-bit VCVTPH2PS variants and would over-tighten the compile-time
// ISA relative to the runtime gate (get_avx512f_status), risking SIGILL
// on AVX-512F CPUs without F16C even on the pure FP32/BF16 paths.
//
// #pragma GCC optimize("no-tree-vectorize") prevents the compiler from
// auto-vectorising scalar-path loops with AVX-512 instructions, which
// would SIGILL on machines without AVX-512 support.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,fma")
#pragma GCC optimize("no-tree-vectorize")
#endif

template <typename SimdTag, typename T1, typename T2>
inline void scale_attn_mask_fusion(
        T1 *a, T2 const *b, int size, T1 *out, T1 val) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    int i = 0;
    if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, float>) {
        const VecF32 vs = Ops::vec_set1(val);
        for (; i + L <= size; i += L) {
            const VecF32 va = Ops::vec_loadu(a + i);
            const VecF32 vb
                    = Ops::vec_loadu(reinterpret_cast<const float *>(b) + i);
            Ops::vec_storeu(out + i, Ops::vec_fmadd(va, vs, vb));
        }
    } else if constexpr (std::is_same_v<T1, float>
            && std::is_same_v<T2, bfloat16_t>) {
        const VecF32 vs = Ops::vec_set1(val);
        const auto *bp = reinterpret_cast<const uint16_t *>(b);
        for (; i + L <= size; i += L) {
            const VecF32 va = Ops::vec_loadu(a + i);
            const VecF32 vb = Ops::vec_mask_bf16_loadu(bp + i);
            Ops::vec_storeu(out + i, Ops::vec_fmadd(va, vs, vb));
        }
    } else if constexpr (std::is_same_v<T1, float>
            && std::is_same_v<T2, float16_t>) {
        const VecF32 vs = Ops::vec_set1(val);
        const auto *bp = reinterpret_cast<const uint16_t *>(b);
        for (; i + L <= size; i += L) {
            const VecF32 va = Ops::vec_loadu(a + i);
            const VecF32 vb = Ops::vec_mask_f16_loadu(bp + i);
            Ops::vec_storeu(out + i, Ops::vec_fmadd(va, vs, vb));
        }
    }
    for (; i < size; ++i) {
        const float bf = mask_elem_at(b, i);
        out[i] = a[i] * val + static_cast<T1>(bf);
    }
}

template <typename SimdTag>
inline void exp_reduce_sum_fusion(
        const accum_t *a, int size, accum_t *out, accum_t &val) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vmb = Ops::vec_set1(val);
    VecF32 vsum = Ops::vec_set1(0.f);
    int i = 0;
    for (; i + L <= size; i += L) {
        const VecF32 x = Ops::vec_loadu(a + i);
        const VecF32 d = Ops::vec_sub(x, vmb);
        const VecF32 e = Ops::vec_exp_u20(d);
        vsum = Ops::vec_add(vsum, e);
        Ops::vec_storeu(out + i, e);
    }
    accum_t tmp_sum = Ops::vec_reduce_sum(vsum);
    for (; i < size; ++i) {
        const accum_t e = std::exp(a[i] - val);
        out[i] = e;
        tmp_sum += e;
    }
    val = tmp_sum;
}

template <typename SimdTag>
inline void fill_stub_f32(float *data, float val, int64_t size) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vv = Ops::vec_set1(val);
    int64_t d = 0;
    for (; d + L <= size; d += L) {
        Ops::vec_storeu(data + d, vv);
    }
    for (; d < size; ++d) {
        data[d] = val;
    }
}

template <typename SimdTag>
inline accum_t row_max(const accum_t *row, int64_t len) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    VecF32 vtmp_max = Ops::vec_set1(-std::numeric_limits<accum_t>::infinity());
    int64_t c = 0;
    for (; c + L <= len; c += L) {
        const VecF32 x = Ops::vec_loadu(row + c);
        vtmp_max = Ops::vec_max(vtmp_max, x);
    }
    accum_t m = Ops::vec_reduce_max(vtmp_max);
    for (; c < len; ++c) {
        m = std::max(m, row[c]);
    }
    return m;
}

template <typename SimdTag>
inline void scale_dst_row(accum_t *row, int64_t len, accum_t exp_tmp) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vs = Ops::vec_set1(exp_tmp);
    int64_t c = 0;
    for (; c + L <= len; c += L) {
        const VecF32 x = Ops::vec_loadu(row + c);
        Ops::vec_storeu(row + c, Ops::vec_mul(x, vs));
    }
    for (; c < len; ++c) {
        row[c] *= exp_tmp;
    }
}

template <typename SimdTag>
inline void vec_f32_scaled_bf16_store(
        bfloat16_t *dst, const accum_t *src, int64_t len, accum_t scale) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vs = Ops::vec_set1(scale);
    int64_t c = 0;
    for (; c + L <= len; c += L) {
        const VecF32 x = Ops::vec_loadu(src + c);
        Ops::vec_bf16_storeu(
                reinterpret_cast<uint16_t *>(&dst[c]), Ops::vec_mul(x, vs));
    }
    for (; c < len; ++c) {
        dst[c] = bfloat16_t(src[c] * scale);
    }
}

template <typename SimdTag>
inline void vec_f32_scaled_f16_store(
        float16_t *dst, const accum_t *src, int64_t len, accum_t scale) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vs = Ops::vec_set1(scale);
    int64_t c = 0;
    for (; c + L <= len; c += L) {
        const VecF32 x = Ops::vec_loadu(src + c);
        Ops::vec_f16_storeu(
                reinterpret_cast<uint16_t *>(&dst[c]), Ops::vec_mul(x, vs));
    }
    for (; c < len; ++c) {
        dst[c] = float16_t(src[c] * scale);
    }
}

template <typename SimdTag, typename scalar_t>
inline void write_scaled_output_row(scalar_t *out_base, int64_t out_stride_d,
        const accum_t *dst_row, int64_t headSize, accum_t sum_reciprocal) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vr = Ops::vec_set1(sum_reciprocal);
    if (out_stride_d == 1) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            int64_t c = 0;
            for (; c + L <= headSize; c += L) {
                const VecF32 x = Ops::vec_loadu(dst_row + c);
                Ops::vec_storeu(out_base + c, Ops::vec_mul(x, vr));
            }
            for (; c < headSize; ++c) {
                out_base[c] = dst_row[c] * sum_reciprocal;
            }
            return;
        } else if constexpr (std::is_same_v<scalar_t, bfloat16_t>) {
            vec_f32_scaled_bf16_store<SimdTag>(
                    out_base, dst_row, headSize, sum_reciprocal);
            return;
        } else if constexpr (std::is_same_v<scalar_t, float16_t>) {
            vec_f32_scaled_f16_store<SimdTag>(
                    out_base, dst_row, headSize, sum_reciprocal);
            return;
        }
    }
    for (int64_t c = 0; c < headSize; ++c) {
        const accum_t v = dst_row[c] * sum_reciprocal;
        out_base[c * out_stride_d] = float_to_scalar<scalar_t>(v);
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

struct TransposedBHSD {
    const void *base;
    int64_t size_b, size_m, size_h, size_d;
    int64_t stride_b, stride_m, stride_h, stride_d;
};

inline TransposedBHSD transpose_bh_sd(const sdpa_flash_cpu_tensor_view &v) {
    TransposedBHSD t;
    t.base = v.data;
    t.size_b = v.size_b;
    t.size_m = v.size_s;
    t.size_h = v.size_h;
    t.size_d = v.size_d;
    t.stride_b = v.stride_b;
    t.stride_m = v.stride_s;
    t.stride_h = v.stride_h;
    t.stride_d = v.stride_d;
    return t;
}

struct NormalizedMask {
    const void *data;
    int64_t m_stride_b;
    int64_t m_stride_h;
    int64_t m_stride_m;
};

inline NormalizedMask normalize_mask(const sdpa_flash_cpu_mask_view &mv,
        int64_t batchSize, int64_t num_head, int64_t qSize, int64_t kvSize) {
    int64_t v0 = 1, v1 = 1, v2, v3;
    int64_t t0 = 0, t1 = 0, t2, t3;

    if (mv.ndim == 2) {
        v2 = mv.sizes[0];
        v3 = mv.sizes[1];
        t2 = mv.strides[0];
        t3 = mv.strides[1];
    } else if (mv.ndim == 4) {
        const int64_t o0 = mv.sizes[0], o1 = mv.sizes[1], o2 = mv.sizes[2],
                      o3 = mv.sizes[3];
        v0 = (o0 == batchSize) ? batchSize : 1;
        v1 = (o1 == num_head) ? num_head : 1;
        v2 = o2;
        v3 = o3;
        t0 = mv.strides[0];
        t1 = mv.strides[1];
        t2 = mv.strides[2];
        t3 = mv.strides[3];
    } else {
        throw std::invalid_argument("normalize_mask: ndim must be 2 or 4");
    }

    auto expand_stride = [&](int64_t vs, int64_t ts, int64_t es) -> int64_t {
        if (es == vs) { return ts; }
        if (vs == 1) { return 0; }
        throw std::invalid_argument("normalize_mask: incompatible mask shape");
    };

    int64_t rs0 = expand_stride(v0, t0, batchSize);
    int64_t rs1 = expand_stride(v1, t1, num_head);
    int64_t rs2 = expand_stride(v2, t2, qSize);
    int64_t rs3 = expand_stride(v3, t3, kvSize);
    (void)rs3;
    SDPA_SA_CHECK(rs3 == 1,
            "normalize_mask: last dim stride must be 1 (contiguous KV)");

    NormalizedMask nm;
    nm.data = mv.data;
    nm.m_stride_b = (batchSize > 1) ? rs0 : 0;
    nm.m_stride_h = (num_head > 1) ? rs1 : 0;
    nm.m_stride_m = rs2;
    return nm;
}

inline float calculate_scale_value(
        std::optional<double> scale, int64_t head_dim) {
    if (scale.has_value()) { return static_cast<float>(scale.value()); }
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// Entry point for the dynamic-INT8 flash path, defined in
// lowoha_sdpa_flash_cpu_int8.cpp.  Performs its own mask-type and SIMD-tag
// dispatch so the INT8 kernel templates stay confined to that TU.
status_t sdpa_flash_cpu_run_int8(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, data_type_t qkv_dt, data_type_t mask_dtype,
        bool is_qk_quant, bool is_pv_quant, bool use_avx512, int num_threads);

// Releases this thread's INT8 scratch pool (owned by the INT8 TU).
void sdpa_flash_cpu_free_int8_scratch();

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // ZENDNNL_LOWOHA_SDPA_FLASH_CPU_COMMON_HPP
