/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

// FP32 / BF16 / FP16 flash-attention path.  The dynamic-INT8 path lives in
// lowoha_sdpa_flash_cpu_int8.cpp and is reached through
// sdpa_flash_cpu_run_int8(); helpers common to both are in
// lowoha_sdpa_flash_cpu_common.hpp.

#include "lowoha_sdpa_flash_cpu_common.hpp"

#include "common/logging.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include <cstdlib>
#include <cstring>

#include <omp.h>

// Suppress GCC's informational ABI-change note for 64-byte vector types
// (__m512).  The note warns that the calling convention for these types
// differs from GCC 4.6 — irrelevant for any modern toolchain.  The
// template helpers below pass __m512 to/from SimdOps<avx512_tag> methods
// that carry ZENDNNL_TARGET("avx512f,..."), which is correct.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

static_assert(simd::SimdOps<simd::avx512_tag>::kFloatLanes >= 1
                && simd::SimdOps<simd::scalar_tag>::kFloatLanes >= 1,
        "simd_ops.hpp must provide at least 1 float lane per tag");

struct scratch_buffer {
    void *ptr = nullptr;
    size_t cap = 0;
    ~scratch_buffer() { free(ptr); }
    scratch_buffer() = default;
    scratch_buffer(const scratch_buffer &) = delete;
    scratch_buffer &operator=(const scratch_buffer &) = delete;
};

static thread_local scratch_buffer g_flash_scratch;

namespace {

inline void *flash_scratch_acquire(size_t bytes) {
    if (g_flash_scratch.cap < bytes) {
        free(g_flash_scratch.ptr);
        g_flash_scratch.ptr = malloc(bytes);
        g_flash_scratch.cap = g_flash_scratch.ptr ? bytes : 0;
    }
    return g_flash_scratch.ptr;
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

template <typename SimdTag, typename T2>
inline void exp_reduce_sum_fusion_to(
        const accum_t *a, int size, T2 *out, accum_t &val) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vmb = Ops::vec_set1(val);
    VecF32 vsum = Ops::vec_set1(0.f);
    int i = 0;
    for (; i + L <= size; i += L) {
        const VecF32 x = Ops::vec_loadu(a + i);
        const VecF32 d = Ops::vec_sub(x, vmb);
        const VecF32 e = Ops::vec_fexp_u20(d);
        vsum = Ops::vec_add(vsum, e);
        if constexpr (std::is_same_v<T2, bfloat16_t>) {
            Ops::vec_bf16_storeu(reinterpret_cast<uint16_t *>(&out[i]), e);
        } else if constexpr (std::is_same_v<T2, float16_t>) {
            Ops::vec_f16_storeu(reinterpret_cast<uint16_t *>(&out[i]), e);
        } else {
            Ops::vec_storeu(reinterpret_cast<float *>(out + i), e);
        }
    }
    accum_t tmp_sum = Ops::vec_reduce_sum(vsum);
    for (; i < size; ++i) {
        const accum_t e = std::exp(a[i] - val);
        out[i] = float_to_scalar<T2>(e);
        tmp_sum += e;
    }
    val = tmp_sum;
}

template <typename SimdTag>
inline void mul_reduce_max_fusion(const accum_t *a, accum_t scale, int size,
        accum_t *out, accum_t &maxv) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    VecF32 vtmp_max = Ops::vec_set1(-std::numeric_limits<accum_t>::infinity());
    const VecF32 vs = Ops::vec_set1(scale);
    int i = 0;
    for (; i + L <= size; i += L) {
        const VecF32 x = Ops::vec_loadu(a + i);
        const VecF32 y = Ops::vec_mul(x, vs);
        vtmp_max = Ops::vec_max(vtmp_max, y);
        Ops::vec_storeu(out + i, y);
    }
    maxv = Ops::vec_reduce_max(vtmp_max);
    for (; i < size; ++i) {
        const accum_t t = a[i] * scale;
        out[i] = t;
        maxv = std::max(maxv, t);
    }
}

// Returns the pointer the next GEMM should consume as its A operand:
//   * F32 path: the FP32 softmax tile in `ptr` (no reduced buffer exists).
//   * BF16 / F16 path: the reduced-precision softmax tile in `ptr2`.
template <typename scalar_t>
inline scalar_t *conditional_data_ptr(scalar_t *ptr, scalar_t *ptr2) {
    SDPA_SA_CHECK(ptr2 == nullptr,
            "conditional_data_ptr: unexpected reduced-fp qk buf");
    return ptr;
}

template <typename scalar_t,
        typename std::enable_if_t<is_reduced_fp_v<scalar_t>, int> = 0>
inline scalar_t *conditional_data_ptr(float *ptr, scalar_t *ptr2) {
    return ptr2;
}

// ---------------------------------------------------------------------------
// Main flash-attention kernel — templated on SimdTag for SIMD dispatch.
// ---------------------------------------------------------------------------

template <typename SimdTag, typename scalar_t, typename mask_t,
        int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention_sa(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query_bh,
        const sdpa_flash_cpu_tensor_view &key_bh,
        const sdpa_flash_cpu_tensor_view &value_bh, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, int num_threads_hint) {
    (void)dropout_p;
    SDPA_SA_CHECK(!dropout_p, "dropout must be 0");

    TransposedBHSD query = transpose_bh_sd(query_bh);
    TransposedBHSD key = transpose_bh_sd(key_bh);
    TransposedBHSD value = transpose_bh_sd(value_bh);
    TransposedBHSD out = transpose_bh_sd(output);

    constexpr bool is_reduced_type = is_reduced_fp_v<scalar_t>;
    accum_t scaling_factor
            = calculate_scale_value(scale, static_cast<int64_t>(query.size_d));

    SDPA_SA_CHECK(query.size_d == value.size_d && key.size_d == value.size_d,
            "Q/K/V head dim mismatch");

    int64_t batchSize = query.size_b;
    int64_t qSize = query.size_m;
    int64_t kvSize = value.size_m;
    int64_t num_head = query.size_h;
    int64_t kv_num_head = key.size_h;
    int64_t headSize = query.size_d;
    SDPA_SA_CHECK(kv_num_head > 0, "K/V head count must be > 0");
    SDPA_SA_CHECK(key.size_h == value.size_h, "K/V head count mismatch");
    SDPA_SA_CHECK(key.size_m == value.size_m, "K/V seq len mismatch");
    SDPA_SA_CHECK(num_head % kv_num_head == 0,
            "Q heads must be divisible by K/V heads");
    int64_t repeat_factor = num_head / kv_num_head;

    // Sliding-window band: query i attends only to keys in
    // [i - win_left, i + win_right]. Disabled by default; when combined with
    // is_causal the band keeps only its left half (j <= i).
    const bool use_window = sliding_window && sliding_window_size > 0;
    const int64_t win_left = use_window ? sliding_window_size - 1 : 0;
    const int64_t win_right
            = (use_window && !is_causal) ? sliding_window_size - 1 : 0;

    NormalizedMask nmask {};
    const void *mask_data_void = nullptr;
    bool has_attn_mask = attn_mask.has_value() && attn_mask->data != nullptr
            && (attn_mask->ndim == 2 || attn_mask->ndim == 4)
            && (attn_mask->ndim == 2
                            ? (attn_mask->sizes[0] * attn_mask->sizes[1] > 0)
                            : (attn_mask->sizes[0] * attn_mask->sizes[1]
                                              * attn_mask->sizes[2]
                                              * attn_mask->sizes[3]
                                      > 0));

    if (has_attn_mask) {
        nmask = normalize_mask(*attn_mask, batchSize, num_head, qSize, kvSize);
        mask_data_void = nmask.data;
    }

    int64_t qStrideB = query.stride_b;
    int64_t qStrideM = query.stride_m;
    int64_t qStrideH = query.stride_h;
    int64_t kStrideB = key.stride_b;
    int64_t kStrideN = key.stride_m;
    int64_t kStrideH = key.stride_h;
    int64_t vStrideB = value.stride_b;
    int64_t vStrideN = value.stride_m;
    int64_t vStrideH = value.stride_h;
    int64_t oStrideB = out.stride_b;
    int64_t oStrideM = out.stride_m;
    int64_t oStrideH = out.stride_h;

    int64_t mStrideB = has_attn_mask ? nmask.m_stride_b : 0;
    int64_t mStrideH = has_attn_mask ? nmask.m_stride_h : 0;
    int64_t mStrideM = has_attn_mask ? nmask.m_stride_m : 0;

    int zen_q_split_size = static_cast<int>(q_split_size);
    if (batchSize > 4) { zen_q_split_size = 512; }
    int64_t qSplitSize = zen_q_split_size > qSize
            ? qSize
            : static_cast<int64_t>(zen_q_split_size);
    int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
    int64_t qSlice = (qSize - 1) / qSplitSize + 1;

    int64_t num_thread
            = (num_threads_hint > 0) ? num_threads_hint : omp_get_max_threads();
    int64_t size_per_thread = qSplitSize * kvSplitSize + qSplitSize + qSplitSize
            + qSplitSize * headSize;

    // Reduced-precision per-thread scratch layout (in elements of scalar_t):
    //   [qk_reduced_data : qSplitSize*kvSplitSize ]                     (bf16/f16)
    // Holds the reduced-precision softmax probabilities that feed the P*V GEMM.
    int64_t reduced_per_thread = 0;
    if constexpr (is_reduced_type) {
        reduced_per_thread = qSplitSize * kvSplitSize;
    }

    const size_t buf_bytes = static_cast<size_t>(num_thread * size_per_thread)
            * sizeof(accum_t);
    const size_t buf_reduced_bytes = is_reduced_type
            ? static_cast<size_t>(num_thread * reduced_per_thread)
                    * sizeof(scalar_t)
            : 0;
    void *scratch = flash_scratch_acquire(buf_bytes + buf_reduced_bytes);
    SDPA_SA_CHECK(scratch != nullptr, "flash scratch allocation failed");

    auto *q_data = static_cast<const scalar_t *>(query.base);
    auto *k_data = static_cast<const scalar_t *>(key.base);
    auto *v_data = static_cast<const scalar_t *>(value.base);
    auto *mask_data = static_cast<const mask_t *>(mask_data_void);
    auto *out_data = static_cast<scalar_t *>(const_cast<void *>(out.base));
    accum_t *buf_data = static_cast<accum_t *>(scratch);
    scalar_t *buf_reduced_data = is_reduced_type
            ? reinterpret_cast<scalar_t *>(
                      static_cast<char *>(scratch) + buf_bytes)
            : nullptr;

    scoped_active_levels active_levels_guard(1);

#pragma omp parallel for schedule(static) num_threads(num_thread)
    for (int64_t z = 0; z < batchSize * num_head * qSlice; ++z) {
        int64_t i = 0, j = 0, k = 0;
        sdpa_data_index_init(z, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = omp_get_thread_num();
        accum_t *buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t *qk_data = buf_ptr;
        accum_t *qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t *qk_sum_data = qk_max_data + qSplitSize;
        accum_t *dst_data = qk_sum_data + qSplitSize;
        scalar_t *qk_reduced_data = is_reduced_type
                ? buf_reduced_data + ompIdx * reduced_per_thread
                : nullptr;

        int64_t m = k * qSplitSize;
        int64_t qBlockSize = std::min(qSplitSize, qSize - m);
        int64_t kv_j = j / repeat_factor;
        fill_stub_f32<SimdTag>(qk_max_data,
                -std::numeric_limits<accum_t>::infinity(), qBlockSize);
        fill_stub_f32<SimdTag>(
                qk_sum_data, static_cast<accum_t>(0), qBlockSize);
        int64_t num_keys
                = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
        // Skipping the kv tiles outside the band is what keeps sliding window
        // O(S*W) instead of O(S^2); n_start is snapped down to a tile boundary.
        int64_t n_start = 0;
        if (use_window) {
            num_keys = std::min(kvSize, m + qBlockSize + win_right);
            n_start = std::max<int64_t>(0, m - win_left) / kvSplitSize
                    * kvSplitSize;
        }

        bool have_kv = false;
        for (int64_t n = n_start; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            int64_t n_gemm = n;
            // Whole-tile skip still misses S=1024 / W=257: every Q tile
            // overlaps both 512-wide KV tiles. Shrink the GEMM to the union
            // of this Q block's band inside the tile (prefix or suffix).
            // Per-row tails stay -inf-filled below.
            if (use_window) {
                const int64_t band_lo = std::max(n, m - win_left);
                const int64_t band_hi
                        = std::min(n + kvBlockSize, m + qBlockSize + win_right);
                if (band_lo >= band_hi) { continue; }
                n_gemm = band_lo;
                kvBlockSize = band_hi - band_lo;
            }
            // GEMM: Q_block @ K_block^T -> qk_data (FP32 accumulator).
            //   alpha=1: softmax scale (1/sqrt(d) or user scale) is fused into
            //            mul_reduce_max_fusion / scale_attn_mask_fusion below.
            //   beta =0: qk_data is reused per kv-tile, no prior contents to keep.
            zendnn_gemm<scalar_t>(qBlockSize, kvBlockSize, headSize, 1.0f,
                    q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                    qStrideM,
                    k_data + i * kStrideB + kv_j * kStrideH + n_gemm * kStrideN,
                    kStrideN, 0.0f, qk_data, kvBlockSize, false, true);

            // The window fill below already enforces the causal right edge
            // (win_right == 0 when is_causal), so the two never both run.
            if (use_window) {
                constexpr accum_t neg_inf
                        = -std::numeric_limits<accum_t>::infinity();
                for (int64_t row = 0; row < qBlockSize; ++row) {
                    const int64_t lo
                            = std::max<int64_t>(0, m + row - win_left - n_gemm);
                    const int64_t hi = std::min(
                            kvBlockSize - 1, m + row + win_right - n_gemm);
                    accum_t *row_ptr = qk_data + row * kvBlockSize;
                    if (hi < lo) {
                        fill_stub_f32<SimdTag>(row_ptr, neg_inf, kvBlockSize);
                        continue;
                    }
                    if (lo > 0) {
                        fill_stub_f32<SimdTag>(row_ptr, neg_inf, lo);
                    }
                    if (hi + 1 < kvBlockSize) {
                        fill_stub_f32<SimdTag>(row_ptr + hi + 1, neg_inf,
                                kvBlockSize - hi - 1);
                    }
                }
            } else if (is_causal && num_keys - n <= kvSplitSize) {
                for (int64_t row = 0; row < qBlockSize; ++row) {
                    int64_t last_col = m + row - n;
                    accum_t *row_ptr = qk_data + row * kvBlockSize;
                    if (last_col < 0) {
                        fill_stub_f32<SimdTag>(row_ptr,
                                -std::numeric_limits<accum_t>::infinity(),
                                kvBlockSize);
                    } else if (last_col + 1 < kvBlockSize) {
                        fill_stub_f32<SimdTag>(row_ptr + last_col + 1,
                                -std::numeric_limits<accum_t>::infinity(),
                                kvBlockSize - last_col - 1);
                    }
                }
            }

            if (has_attn_mask) {
                for (int64_t row = 0; row < qBlockSize; ++row) {
                    scale_attn_mask_fusion<SimdTag>(qk_data + row * kvBlockSize,
                            mask_data + i * mStrideB + j * mStrideH
                                    + (m + row) * mStrideM + n_gemm,
                            static_cast<int>(kvBlockSize),
                            qk_data + row * kvBlockSize, scaling_factor);
                }
            }

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
                if (has_attn_mask) {
                    tmp_max = row_max<SimdTag>(
                            qk_data + row * kvBlockSize, kvBlockSize);
                } else {
                    mul_reduce_max_fusion<SimdTag>(qk_data + row * kvBlockSize,
                            scaling_factor, static_cast<int>(kvBlockSize),
                            qk_data + row * kvBlockSize, tmp_max);
                }
                tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row]
                                                     : tmp_max;
                scalar_t *qk_row
                        = conditional_data_ptr(qk_data, qk_reduced_data)
                        + row * kvBlockSize;
                // A fully masked tile has max=-inf. Skip exponentiation because
                // subtracting that max from its -inf scores would produce NaN;
                // zero probabilities make this tile contribute nothing.
                if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                    std::fill_n(qk_row, kvBlockSize, scalar_t(0));
                    continue;
                }
                tmp_sum = tmp_max;
                if constexpr (is_reduced_type) {
                    exp_reduce_sum_fusion_to<SimdTag>(
                            qk_data + row * kvBlockSize,
                            static_cast<int>(kvBlockSize), qk_row, tmp_sum);
                } else {
                    exp_reduce_sum_fusion<SimdTag>(qk_data + row * kvBlockSize,
                            static_cast<int>(kvBlockSize),
                            qk_data + row * kvBlockSize, tmp_sum);
                }
                exp_tmp = std::exp(qk_max_data[row] - tmp_max);
                qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
                qk_max_data[row] = tmp_max;
                if (have_kv) {
                    scale_dst_row<SimdTag>(
                            dst_data + row * headSize, headSize, exp_tmp);
                }
            }
            // GEMM: softmax_block @ V_block -> dst_data (online-softmax accumulator).
            //   alpha=1: prior dst rows already rescaled by exp(prev_max - new_max)
            //            via scale_dst_row above.
            //   beta = 0 on the first live tile; 1 accumulates later tiles into
            //            the FP32 dst_data scratch directly.
            zendnn_gemm<scalar_t>(qBlockSize, headSize, kvBlockSize, 1.0f,
                    conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
                    v_data + i * vStrideB + kv_j * vStrideH + n_gemm * vStrideN,
                    vStrideN, have_kv ? 1.0f : 0.0f, dst_data, headSize, false,
                    false);
            have_kv = true;
        }

        // A q block whose window lies entirely outside [0, kvSize) runs no kv
        // tile at all, so the P*V GEMM never initialised the accumulator.
        if (!have_kv) {
            fill_stub_f32<SimdTag>(
                    dst_data, static_cast<accum_t>(0), qBlockSize * headSize);
        }

        for (int64_t row = 0; row < qBlockSize; ++row) {
            // A row that remained fully masked across every KV tile retains
            // max=-inf and sum=0. Use neutral normalization sentinels so the
            // zero output accumulator stays zero instead of producing NaN.
            if (qk_max_data[row] == -std::numeric_limits<accum_t>::infinity()) {
                qk_max_data[row] = 0;
            }
            if (qk_sum_data[row] == 0) { qk_sum_data[row] = 1; }
            const accum_t sum_reciprocal = 1 / qk_sum_data[row];
            write_scaled_output_row<SimdTag, scalar_t>(out_data + i * oStrideB
                            + j * oStrideH + m * oStrideM + row * oStrideM,
                    out.stride_d, dst_data + row * headSize, headSize,
                    sum_reciprocal);
        }
    }
}

template <typename SimdTag, typename input_type, typename attention_mask>
void flash_attention_kernel_sa_dispatch(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, int num_threads) {
    int64_t q_seq_len = query.size_s;
    if (q_seq_len >= 768) {
        cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 256, 512>(
                output, query, key, value, dropout_p, is_causal, sliding_window,
                sliding_window_size, attn_mask, scale, num_threads);
    } else if (q_seq_len >= 192) {
        cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 64, 512>(
                output, query, key, value, dropout_p, is_causal, sliding_window,
                sliding_window_size, attn_mask, scale, num_threads);
    } else {
        cpu_flash_attention_sa<SimdTag, input_type, attention_mask, 32, 512>(
                output, query, key, value, dropout_p, is_causal, sliding_window,
                sliding_window_size, attn_mask, scale, num_threads);
    }
}

} // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void sdpa_flash_cpu_free_scratch() {
    free(g_flash_scratch.ptr);
    g_flash_scratch.ptr = nullptr;
    g_flash_scratch.cap = 0;
    sdpa_flash_cpu_free_int8_scratch();
}

status_t sdpa_flash_cpu_run_internal(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        const sdpa_flash_cpu_mask_view *mask, const double *scale_opt,
        data_type_t qkv_dt, data_type_t mask_dtype, bool is_qk_quant,
        bool is_pv_quant, int num_threads) {
    try {
        std::optional<sdpa_flash_cpu_mask_view> mopt;
        if (mask != nullptr && mask->data != nullptr) { mopt = *mask; }
        std::optional<double> scale;
        if (scale_opt != nullptr) { scale = *scale_opt; }
        if (dropout_p != 0.0) {
            log_error("sdpa_flash_cpu: dropout must be 0");
            return status_t::failure;
        }

        // FP16 requires AVX512-FP16, matching the matmul ISA gate in
        // lowoha_matmul.cpp. Reject early so callers get a stable
        // status_t::isa_unsupported instead of a silent precision regression.
        if (qkv_dt == data_type_t::f16
                && !zendnnl::common::zendnnl_platform_info()
                            .get_avx512_f16_status()) {
            log_error("sdpa_flash_cpu: FP16 requires AVX512-FP16");
            return status_t::isa_unsupported;
        }
        const bool any_int8_quant = is_qk_quant || is_pv_quant;
        // VNNI is required by the s8xs8 / u8xs8 matmuls; F, BW and VL are
        // required by group_dynamic_quant (lowoha_reorder.cpp).
        if (any_int8_quant
                && (!zendnnl::common::zendnnl_platform_info()
                                .get_avx512f_status()
                        || !zendnnl::common::zendnnl_platform_info()
                                    .get_avx512_bw_vl_status()
                        || !zendnnl::common::zendnnl_platform_info()
                                    .get_avx512_vnni_status())) {
            log_error(
                    "sdpa_flash_cpu: dynamic INT8 path requires AVX512-F, "
                    "AVX512-BW, AVX512-VL, and AVX512-VNNI");
            return status_t::isa_unsupported;
        }

        // Runtime SIMD dispatch: AVX-512 when available, scalar fallback
        // otherwise.
        const bool use_avx512
                = zendnnl::common::zendnnl_platform_info().get_avx512f_status();
        apilog_info(use_avx512 ? "sdpa_flash_cpu: using AVX-512 SIMD"
                               : "sdpa_flash_cpu: using scalar SIMD");

        if (any_int8_quant) {
            return sdpa_flash_cpu_run_int8(output, query, key, value, dropout_p,
                    is_causal, sliding_window, sliding_window_size, mopt, scale,
                    qkv_dt, mask_dtype, is_qk_quant, is_pv_quant, use_avx512,
                    num_threads);
        }

        auto run = [&](auto simd_tag) -> status_t {
            using Tag = decltype(simd_tag);
            // No-mask and f32-mask paths share the same mask_t = float
            // instantiation; mopt is already std::nullopt when no mask is
            // present.
            if (qkv_dt == data_type_t::f32) {
                flash_attention_kernel_sa_dispatch<Tag, float, float>(output,
                        query, key, value, dropout_p, is_causal, sliding_window,
                        sliding_window_size, mopt, scale, num_threads);
            } else if (qkv_dt == data_type_t::bf16) {
                if (!mopt.has_value() || mask_dtype == data_type_t::f32) {
                    flash_attention_kernel_sa_dispatch<Tag, bfloat16_t, float>(
                            output, query, key, value, dropout_p, is_causal,
                            sliding_window, sliding_window_size, mopt, scale,
                            num_threads);
                } else {
                    flash_attention_kernel_sa_dispatch<Tag, bfloat16_t,
                            bfloat16_t>(output, query, key, value, dropout_p,
                            is_causal, sliding_window, sliding_window_size,
                            mopt, scale, num_threads);
                }
            } else if (qkv_dt == data_type_t::f16) {
                if (!mopt.has_value() || mask_dtype == data_type_t::f32) {
                    flash_attention_kernel_sa_dispatch<Tag, float16_t, float>(
                            output, query, key, value, dropout_p, is_causal,
                            sliding_window, sliding_window_size, mopt, scale,
                            num_threads);
                } else {
                    flash_attention_kernel_sa_dispatch<Tag, float16_t,
                            float16_t>(output, query, key, value, dropout_p,
                            is_causal, sliding_window, sliding_window_size,
                            mopt, scale, num_threads);
                }
            } else {
                log_error("sdpa_flash_cpu: unsupported Q/K/V dtype");
                throw std::invalid_argument("unsupported Q/K/V dtype");
            }
            return status_t::success;
        };

        return use_avx512 ? run(simd::avx512_tag {}) : run(simd::scalar_tag {});
    } catch (const std::invalid_argument &e) {
        log_error("sdpa_flash_cpu: invalid argument (check mask/shapes): ",
                e.what());
        return status_t::failure;
    } catch (const std::exception &e) {
        log_error("sdpa_flash_cpu: execution failed: ", e.what());
        return status_t::failure;
    }
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
