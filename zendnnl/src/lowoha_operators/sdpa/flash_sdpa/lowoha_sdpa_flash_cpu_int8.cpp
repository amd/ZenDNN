/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

// Dynamic-INT8 flash-attention path.  Q/K are quantized per token and V per
// channel, the softmax tile is quantized to U8, and the QK / PV matmuls run
// through the INT8 AOCL kernels.  The FP32 / BF16 / FP16 path lives in
// lowoha_sdpa_flash_cpu.cpp; helpers common to both are in
// lowoha_sdpa_flash_cpu_common.hpp.

#include "lowoha_sdpa_flash_cpu_common.hpp"
#include "lowoha_sdpa_int8_utils.hpp"

#include "common/config_params.hpp"
#include "common/logging.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

struct int8_scratch_buffers {
    std::vector<int8_t> q_s8;
    std::vector<int8_t> k_s8;
    std::vector<int8_t> v_s8;
    std::vector<float> q_scales;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
    std::vector<float> f32;
    std::vector<zendnnl::common::bfloat16_t> p_bf16;
    std::vector<uint8_t> p_u8;

    void release() {
        std::vector<int8_t>().swap(q_s8);
        std::vector<int8_t>().swap(k_s8);
        std::vector<int8_t>().swap(v_s8);
        std::vector<float>().swap(q_scales);
        std::vector<float>().swap(k_scales);
        std::vector<float>().swap(v_scales);
        std::vector<float>().swap(f32);
        std::vector<zendnnl::common::bfloat16_t>().swap(p_bf16);
        std::vector<uint8_t>().swap(p_u8);
    }
};

static thread_local int8_scratch_buffers g_int8_flash_scratch;

namespace {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,fma")
#pragma GCC optimize("no-tree-vectorize")
#endif

template <typename SimdTag>
inline void exp_reduce_sum_fusion_to_u8(
        const accum_t *a, int size, uint8_t *out, accum_t &val) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    const VecF32 vmb = Ops::vec_set1(val);
    const VecF32 vu8_scale = Ops::vec_set1(255.0f);
    VecF32 vsum = Ops::vec_set1(0.0f);
    int i = 0;
    for (; i + L <= size; i += L) {
        const VecF32 x = Ops::vec_loadu(a + i);
        const VecF32 e = Ops::vec_exp_u20(Ops::vec_sub(x, vmb));
        vsum = Ops::vec_add(vsum, e);
        Ops::vec_u8_storeu(out + i, Ops::vec_mul(e, vu8_scale));
    }
    accum_t tmp_sum = Ops::vec_reduce_sum(vsum);
    for (; i < size; ++i) {
        const accum_t e = std::exp(a[i] - val);
        const accum_t quantized = std::nearbyint(
                std::max(accum_t(0), std::min(e * 255.0f, accum_t(255))));
        out[i] = static_cast<uint8_t>(quantized);
        tmp_sum += e;
    }
    val = tmp_sum;
}

template <typename SimdTag>
inline void add_f32_inplace(float *dst, const float *src, int64_t size) {
    using Ops = simd::SimdOps<SimdTag>;
    using VecF32 = typename Ops::VecF32;
    const int L = Ops::kFloatLanes;
    int64_t i = 0;
    for (; i + L <= size; i += L) {
        const VecF32 vd = Ops::vec_loadu(dst + i);
        const VecF32 vs = Ops::vec_loadu(src + i);
        Ops::vec_storeu(dst + i, Ops::vec_add(vd, vs));
    }
    for (; i < size; ++i) {
        dst[i] += src[i];
    }
}

// QkInt8 selects an INT8 QK matmul over per-token quantized Q/K; when false
// QK stays on the BF16 GEMM and the attention factor rides in its alpha.
// PvInt8 selects a U8 x S8 PV matmul over the per-channel quantized V.
template <typename SimdTag, typename mask_t, int64_t q_split_size,
        int64_t kv_split_size, bool QkInt8, bool PvInt8>
status_t cpu_flash_attention_int8_dq(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query_bh,
        const sdpa_flash_cpu_tensor_view &key_bh,
        const sdpa_flash_cpu_tensor_view &value_bh, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, int num_threads_hint) {
    SDPA_SA_CHECK(!dropout_p, "dropout must be 0");

    TransposedBHSD query = transpose_bh_sd(query_bh);
    TransposedBHSD key = transpose_bh_sd(key_bh);
    TransposedBHSD value = transpose_bh_sd(value_bh);
    TransposedBHSD out = transpose_bh_sd(output);

    SDPA_SA_CHECK(query.size_d == value.size_d && key.size_d == value.size_d,
            "Q/K/V head dim mismatch");

    const int64_t batchSize = query.size_b;
    const int64_t qSize = query.size_m;
    const int64_t kvSize = value.size_m;
    const int64_t num_head = query.size_h;
    const int64_t kv_num_head = key.size_h;
    const int64_t headSize = query.size_d;
    SDPA_SA_CHECK(kv_num_head > 0, "K/V head count must be > 0");
    SDPA_SA_CHECK(key.size_h == value.size_h, "K/V head count mismatch");
    SDPA_SA_CHECK(key.size_m == value.size_m, "K/V seq len mismatch");
    SDPA_SA_CHECK(num_head % kv_num_head == 0,
            "Q heads must be divisible by K/V heads");
    const int64_t repeat_factor = num_head / kv_num_head;
    const accum_t scaling_factor = calculate_scale_value(scale, headSize);
    constexpr bool use_int8_pv = PvInt8;

    // Sliding-window band: query i attends only to keys in
    // [i - win_left, i + win_right]. Disabled by default; when combined with
    // is_causal the band keeps only its left half (j <= i).
    const bool use_window = sliding_window && sliding_window_size > 0;
    const int64_t win_left = use_window ? sliding_window_size - 1 : 0;
    const int64_t win_right
            = (use_window && !is_causal) ? sliding_window_size - 1 : 0;

    NormalizedMask nmask {};
    const void *mask_data_void = nullptr;
    const bool has_attn_mask = attn_mask.has_value()
            && attn_mask->data != nullptr
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

    const int64_t qStrideB = query.stride_b;
    const int64_t qStrideM = query.stride_m;
    const int64_t qStrideH = query.stride_h;
    const int64_t kStrideB = key.stride_b;
    const int64_t kStrideN = key.stride_m;
    const int64_t kStrideH = key.stride_h;
    const int64_t vStrideB = value.stride_b;
    const int64_t vStrideN = value.stride_m;
    const int64_t vStrideH = value.stride_h;
    const int64_t oStrideB = out.stride_b;
    const int64_t oStrideM = out.stride_m;
    const int64_t oStrideH = out.stride_h;
    const int64_t mStrideB = has_attn_mask ? nmask.m_stride_b : 0;
    const int64_t mStrideH = has_attn_mask ? nmask.m_stride_h : 0;
    const int64_t mStrideM = has_attn_mask ? nmask.m_stride_m : 0;

    int64_t qSplitSize = std::min<int64_t>(q_split_size, qSize);
    if (batchSize > 4) { qSplitSize = std::min<int64_t>(512, qSize); }
    const int64_t kvSplitSize = std::min<int64_t>(kv_split_size, kvSize);
    const int64_t qSlice = (qSize - 1) / qSplitSize + 1;
    const int num_thread
            = (num_threads_hint > 0) ? num_threads_hint : omp_get_max_threads();

    const auto *q_data = static_cast<const bfloat16_t *>(query.base);
    const auto *k_data = static_cast<const bfloat16_t *>(key.base);
    const auto *v_data = static_cast<const bfloat16_t *>(value.base);
    const auto *mask_data = static_cast<const mask_t *>(mask_data_void);
    auto *out_data = static_cast<bfloat16_t *>(const_cast<void *>(out.base));

    // Quantize all Q and K heads together, retaining one scale per token.
    const size_t q_head_elems
            = static_cast<size_t>(qSize) * static_cast<size_t>(headSize);
    const size_t kv_head_elems
            = static_cast<size_t>(kvSize) * static_cast<size_t>(headSize);
    const size_t q_head_count
            = static_cast<size_t>(batchSize) * static_cast<size_t>(num_head);
    const size_t kv_head_count
            = static_cast<size_t>(batchSize) * static_cast<size_t>(kv_num_head);

    // Checked before any scratch is sized. The grouped-quant API takes its
    // extents as int, so this guards the static_cast<int> narrowing below;
    // the products above cannot overflow size_t because
    // validate_flash_sdpa_inputs already rejects non-positive dims.
    SDPA_SA_CHECK(qSize <= std::numeric_limits<int>::max()
                    && kvSize <= std::numeric_limits<int>::max()
                    && headSize <= std::numeric_limits<int>::max(),
            "INT8 grouped quantization dimensions exceed INT_MAX");

    auto &q_s8 = g_int8_flash_scratch.q_s8;
    auto &k_s8 = g_int8_flash_scratch.k_s8;
    auto &v_s8 = g_int8_flash_scratch.v_s8;
    auto &q_scales = g_int8_flash_scratch.q_scales;
    auto &k_scales = g_int8_flash_scratch.k_scales;
    auto &v_scales = g_int8_flash_scratch.v_scales;
    if constexpr (QkInt8) {
        ensure_scratch_size(q_s8, q_head_count * q_head_elems);
        ensure_scratch_size(k_s8, kv_head_count * kv_head_elems);
        ensure_scratch_size(
                q_scales, q_head_count * static_cast<size_t>(qSize));
        ensure_scratch_size(
                k_scales, kv_head_count * static_cast<size_t>(kvSize));
    }
    if (use_int8_pv) {
        ensure_scratch_size(v_s8, kv_head_count * kv_head_elems);
        ensure_scratch_size(
                v_scales, kv_head_count * static_cast<size_t>(headSize));
    }

    scoped_active_levels active_levels_guard(1);

    if constexpr (QkInt8) {
        const size_t quant_op_count = q_head_count + kv_head_count;
        std::vector<const void *> quant_src;
        std::vector<int> quant_rows;
        std::vector<int> quant_cols;
        std::vector<std::vector<int64_t>> quant_src_strides;
        std::vector<void *> quant_dst;
        std::vector<void *> quant_scales;
        quant_src.reserve(quant_op_count);
        quant_rows.reserve(quant_op_count);
        quant_cols.reserve(quant_op_count);
        quant_src_strides.reserve(quant_op_count);
        quant_dst.reserve(quant_op_count);
        quant_scales.reserve(quant_op_count);

        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t h = 0; h < num_head; ++h) {
                const size_t head_index = static_cast<size_t>(b * num_head + h);
                quant_src.push_back(q_data + b * qStrideB + h * qStrideH);
                quant_rows.push_back(static_cast<int>(qSize));
                quant_cols.push_back(static_cast<int>(headSize));
                quant_src_strides.push_back({qStrideM, 1});
                quant_dst.push_back(q_s8.data() + head_index * q_head_elems);
                quant_scales.push_back(q_scales.data()
                        + head_index * static_cast<size_t>(qSize));
            }
        }
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t h = 0; h < kv_num_head; ++h) {
                const size_t head_index
                        = static_cast<size_t>(b * kv_num_head + h);
                quant_src.push_back(k_data + b * kStrideB + h * kStrideH);
                quant_rows.push_back(static_cast<int>(kvSize));
                quant_cols.push_back(static_cast<int>(headSize));
                quant_src_strides.push_back({kStrideN, 1});
                quant_dst.push_back(k_s8.data() + head_index * kv_head_elems);
                quant_scales.push_back(k_scales.data()
                        + head_index * static_cast<size_t>(kvSize));
            }
        }

        const status_t quant_status = sdpa_group_quantize_bf16_s8_per_token(
                quant_src, quant_rows, quant_cols, quant_src_strides, quant_dst,
                quant_scales, num_thread);
        if (quant_status != status_t::success) {
            log_error("sdpa_flash_cpu: dynamic Q/K quantization failed");
            return status_t::failure;
        }
    }

    if (use_int8_pv) {
        std::vector<const void *> v_quant_src;
        std::vector<int> v_quant_rows;
        std::vector<int> v_quant_cols;
        std::vector<std::vector<int64_t>> v_quant_src_strides;
        std::vector<void *> v_quant_dst;
        std::vector<void *> v_quant_scales;
        v_quant_src.reserve(kv_head_count);
        v_quant_rows.reserve(kv_head_count);
        v_quant_cols.reserve(kv_head_count);
        v_quant_src_strides.reserve(kv_head_count);
        v_quant_dst.reserve(kv_head_count);
        v_quant_scales.reserve(kv_head_count);
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t h = 0; h < kv_num_head; ++h) {
                const size_t head_index
                        = static_cast<size_t>(b * kv_num_head + h);
                v_quant_src.push_back(v_data + b * vStrideB + h * vStrideH);
                v_quant_rows.push_back(static_cast<int>(kvSize));
                v_quant_cols.push_back(static_cast<int>(headSize));
                v_quant_src_strides.push_back({vStrideN, 1});
                v_quant_dst.push_back(v_s8.data() + head_index * kv_head_elems);
                v_quant_scales.push_back(v_scales.data()
                        + head_index * static_cast<size_t>(headSize));
            }
        }

        const status_t v_quant_status = sdpa_group_quantize_bf16_s8_per_channel(
                v_quant_src, v_quant_rows, v_quant_cols, v_quant_src_strides,
                v_quant_dst, v_quant_scales, num_thread);
        if (v_quant_status != status_t::success) {
            log_error("sdpa_flash_cpu: dynamic V quantization failed");
            return status_t::failure;
        }
    }

    if constexpr (QkInt8) {
        if (scaling_factor != accum_t(1)) {
            // QK dequantization multiplies each row by its Q scale. Carry the
            // attention factor there so every score tile avoids a scaling pass.
            const size_t q_scale_count
                    = q_head_count * static_cast<size_t>(qSize);
            for (size_t index = 0; index < q_scale_count; ++index) {
                q_scales[index] *= scaling_factor;
            }
        }
    }

    const int64_t qk_tile_capacity = qSplitSize * kvSplitSize;
    const int64_t pv_tile_capacity = use_int8_pv ? qSplitSize * headSize : 0;
    const int64_t size_per_thread = qk_tile_capacity + qSplitSize + qSplitSize
            + qSplitSize * headSize + pv_tile_capacity;
    auto &f32_scratch = g_int8_flash_scratch.f32;
    auto &p_bf16 = g_int8_flash_scratch.p_bf16;
    auto &p_u8 = g_int8_flash_scratch.p_u8;
    ensure_scratch_size(
            f32_scratch, static_cast<size_t>(num_thread * size_per_thread));
    if (use_int8_pv) {
        ensure_scratch_size(
                p_u8, static_cast<size_t>(num_thread * qk_tile_capacity));
    } else {
        ensure_scratch_size(
                p_bf16, static_cast<size_t>(num_thread * qk_tile_capacity));
    }

    constexpr float probability_scale = 1.0f / 255.0f;
    std::atomic<bool> execute_failed {false};
#pragma omp parallel for schedule(static) num_threads(num_thread)
    for (int64_t z = 0; z < batchSize * num_head * qSlice; ++z) {
        if (execute_failed.load(std::memory_order_relaxed)) { continue; }

        int64_t i = 0, j = 0, k = 0;
        sdpa_data_index_init(z, i, batchSize, j, num_head, k, qSlice);
        const int ompIdx = omp_get_thread_num();
        accum_t *buf_ptr = f32_scratch.data() + ompIdx * size_per_thread;
        accum_t *qk_data = buf_ptr;
        accum_t *qk_max_data = qk_data + qk_tile_capacity;
        accum_t *qk_sum_data = qk_max_data + qSplitSize;
        accum_t *dst_data = qk_sum_data + qSplitSize;
        accum_t *pv_data
                = use_int8_pv ? dst_data + qSplitSize * headSize : nullptr;
        bfloat16_t *p_bf16_tile = use_int8_pv ? nullptr
                                              : p_bf16.data()
                        + static_cast<size_t>(ompIdx * qk_tile_capacity);
        uint8_t *p_u8_tile = use_int8_pv
                ? p_u8.data() + static_cast<size_t>(ompIdx * qk_tile_capacity)
                : nullptr;

        const int64_t m = k * qSplitSize;
        const int64_t qBlockSize = std::min(qSplitSize, qSize - m);
        const int64_t kv_j = j / repeat_factor;
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

        const size_t q_head_index = static_cast<size_t>(i * num_head + j);
        const size_t kv_head_index
                = static_cast<size_t>(i * kv_num_head + kv_j);
        const int8_t *q_tile = QkInt8
                ? q_s8.data() + q_head_index * q_head_elems
                        + static_cast<size_t>(m * headSize)
                : nullptr;
        const float *q_tile_scales = QkInt8
                ? q_scales.data() + q_head_index * static_cast<size_t>(qSize)
                        + static_cast<size_t>(m)
                : nullptr;
        const int8_t *k_head = QkInt8
                ? k_s8.data() + kv_head_index * kv_head_elems
                : nullptr;
        const float *k_head_scales = QkInt8
                ? k_scales.data() + kv_head_index * static_cast<size_t>(kvSize)
                : nullptr;
        const bfloat16_t *q_bf16_tile = QkInt8
                ? nullptr
                : q_data + i * qStrideB + j * qStrideH + m * qStrideM;
        const bfloat16_t *k_bf16_head
                = QkInt8 ? nullptr : k_data + i * kStrideB + kv_j * kStrideH;
        const int8_t *v_head = use_int8_pv
                ? v_s8.data() + kv_head_index * kv_head_elems
                : nullptr;
        const float *v_head_scales = use_int8_pv ? v_scales.data()
                        + kv_head_index * static_cast<size_t>(headSize)
                                                 : nullptr;

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
            status_t st = status_t::success;
            if constexpr (QkInt8) {
                const int8_t *k_tile
                        = k_head + static_cast<size_t>(n_gemm * headSize);
                st = sdpa_qk_int8_matmul(q_tile, k_tile, qk_data,
                        static_cast<int>(qBlockSize),
                        static_cast<int>(kvBlockSize),
                        static_cast<int>(headSize), q_tile_scales,
                        k_head_scales + n_gemm);
                if (st != status_t::success) {
                    execute_failed.store(true, std::memory_order_relaxed);
                    break;
                }
            } else {
                // No Q scale to carry the attention factor, so fold it into
                // the GEMM alpha; everything downstream then matches the
                // INT8-QK path, which pre-scales q_scales.
                zendnn_gemm<bfloat16_t>(qBlockSize, kvBlockSize, headSize,
                        scaling_factor, q_bf16_tile, qStrideM,
                        k_bf16_head + n_gemm * kStrideN, kStrideN, 0.0f,
                        qk_data, kvBlockSize, false, true);
            }

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
                    const int64_t last_col = m + row - n;
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
                            qk_data + row * kvBlockSize, accum_t(1));
                }
            }

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
                // The attention factor is already folded into this row's Q
                // dequantization scale, so only the maximum reduction remains.
                tmp_max = row_max<SimdTag>(
                        qk_data + row * kvBlockSize, kvBlockSize);
                tmp_max = std::max(qk_max_data[row], tmp_max);
                accum_t *qk_row = qk_data + row * kvBlockSize;
                uint8_t *p_u8_row
                        = use_int8_pv ? p_u8_tile + row * kvBlockSize : nullptr;
                if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                    if (use_int8_pv) {
                        std::fill_n(p_u8_row, kvBlockSize, uint8_t(0));
                    } else {
                        std::fill_n(qk_row, kvBlockSize, accum_t(0));
                    }
                    continue;
                }
                tmp_sum = tmp_max;
                if (use_int8_pv) {
                    exp_reduce_sum_fusion_to_u8<SimdTag>(qk_row,
                            static_cast<int>(kvBlockSize), p_u8_row, tmp_sum);
                } else {
                    exp_reduce_sum_fusion<SimdTag>(qk_row,
                            static_cast<int>(kvBlockSize), qk_row, tmp_sum);
                }
                exp_tmp = std::exp(qk_max_data[row] - tmp_max);
                qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
                qk_max_data[row] = tmp_max;
                if (have_kv) {
                    scale_dst_row<SimdTag>(
                            dst_data + row * headSize, headSize, exp_tmp);
                }
            }

            if (use_int8_pv) {
                accum_t *pv_output = have_kv ? pv_data : dst_data;
                st = sdpa_pv_int8_matmul(p_u8_tile,
                        v_head + static_cast<size_t>(n_gemm * headSize),
                        pv_output, static_cast<int>(qBlockSize),
                        static_cast<int>(headSize),
                        static_cast<int>(kvBlockSize), &probability_scale,
                        v_head_scales);
                if (st != status_t::success) {
                    execute_failed.store(true, std::memory_order_relaxed);
                    break;
                }
                if (have_kv) {
                    add_f32_inplace<SimdTag>(
                            dst_data, pv_data, qBlockSize * headSize);
                }
            } else {
                // QK-only mode keeps probability×V on the BF16 path.
                vec_f32_scaled_bf16_store<SimdTag>(p_bf16_tile, qk_data,
                        qBlockSize * kvBlockSize, accum_t(1));
                zendnn_gemm<bfloat16_t>(qBlockSize, headSize, kvBlockSize, 1.0f,
                        p_bf16_tile, kvBlockSize,
                        v_data + i * vStrideB + kv_j * vStrideH
                                + n_gemm * vStrideN,
                        vStrideN, have_kv ? 1.0f : 0.0f, dst_data, headSize,
                        false, false);
            }
            have_kv = true;
        }

        if (execute_failed.load(std::memory_order_relaxed)) { continue; }
        // A q block whose window lies entirely outside [0, kvSize) runs no kv
        // tile at all, so the P*V GEMM never initialised the accumulator.
        if (!have_kv) {
            fill_stub_f32<SimdTag>(
                    dst_data, static_cast<accum_t>(0), qBlockSize * headSize);
        }
        for (int64_t row = 0; row < qBlockSize; ++row) {
            if (qk_max_data[row] == -std::numeric_limits<accum_t>::infinity()) {
                qk_max_data[row] = 0;
            }
            if (qk_sum_data[row] == 0) { qk_sum_data[row] = 1; }
            const accum_t sum_reciprocal = 1 / qk_sum_data[row];
            write_scaled_output_row<SimdTag, bfloat16_t>(out_data + i * oStrideB
                            + j * oStrideH + m * oStrideM + row * oStrideM,
                    out.stride_d, dst_data + row * headSize, headSize,
                    sum_reciprocal);
        }
    }

    if (execute_failed.load(std::memory_order_relaxed)) {
        log_error("sdpa_flash_cpu: dynamic INT8 matmul/quantization failed");
        return status_t::failure;
    }
    return status_t::success;
}

// Selects the q/kv tile split for the resolved <QkInt8, PvInt8> mode.
template <typename SimdTag, typename attention_mask, bool QkInt8, bool PvInt8>
status_t flash_attention_int8_kernel_dispatch(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, int num_threads) {
    const int64_t q_seq_len = query.size_s;
    if (q_seq_len >= 768) {
        return cpu_flash_attention_int8_dq<SimdTag, attention_mask, 256, 512,
                QkInt8, PvInt8>(output, query, key, value, dropout_p, is_causal,
                sliding_window, sliding_window_size, attn_mask, scale,
                num_threads);
    }
    if (q_seq_len >= 192) {
        return cpu_flash_attention_int8_dq<SimdTag, attention_mask, 64, 512,
                QkInt8, PvInt8>(output, query, key, value, dropout_p, is_causal,
                sliding_window, sliding_window_size, attn_mask, scale,
                num_threads);
    }
    return cpu_flash_attention_int8_dq<SimdTag, attention_mask, 32, 512, QkInt8,
            PvInt8>(output, query, key, value, dropout_p, is_causal,
            sliding_window, sliding_window_size, attn_mask, scale, num_threads);
}

// Turns the two runtime mode flags into one of the three INT8 template
// instantiations.  (false, false) never reaches here -- the caller routes it
// to the FP path.
template <typename SimdTag, typename attention_mask>
status_t flash_attention_int8_mode_dispatch(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, bool is_qk_quant, bool is_pv_quant,
        int num_threads) {
    if (is_qk_quant && is_pv_quant) {
        return flash_attention_int8_kernel_dispatch<SimdTag, attention_mask,
                true, true>(output, query, key, value, dropout_p, is_causal,
                sliding_window, sliding_window_size, attn_mask, scale,
                num_threads);
    }
    if (is_qk_quant) {
        return flash_attention_int8_kernel_dispatch<SimdTag, attention_mask,
                true, false>(output, query, key, value, dropout_p, is_causal,
                sliding_window, sliding_window_size, attn_mask, scale,
                num_threads);
    }
    return flash_attention_int8_kernel_dispatch<SimdTag, attention_mask, false,
            true>(output, query, key, value, dropout_p, is_causal,
            sliding_window, sliding_window_size, attn_mask, scale, num_threads);
}

} // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void sdpa_flash_cpu_free_int8_scratch() {
    g_int8_flash_scratch.release();
}

status_t sdpa_flash_cpu_run_int8(const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, double dropout_p,
        bool is_causal, bool sliding_window, int64_t sliding_window_size,
        std::optional<sdpa_flash_cpu_mask_view> attn_mask,
        std::optional<double> scale, data_type_t qkv_dt, data_type_t mask_dtype,
        bool is_qk_quant, bool is_pv_quant, bool use_avx512, int num_threads) {
    if (qkv_dt != data_type_t::bf16) {
        log_error("sdpa_flash_cpu: dynamic INT8 path requires BF16 Q/K/V");
        return status_t::unimplemented;
    }
    if (!is_qk_quant && !is_pv_quant) {
        log_error("sdpa_flash_cpu: INT8 path needs is_qk_quant or is_pv_quant");
        return status_t::failure;
    }

    // No-mask and f32-mask share the mask_t = float instantiation;
    // attn_mask is already std::nullopt when no mask is present.
    auto run = [&](auto simd_tag) -> status_t {
        using Tag = decltype(simd_tag);
        if (!attn_mask.has_value() || mask_dtype == data_type_t::f32) {
            return flash_attention_int8_mode_dispatch<Tag, float>(output, query,
                    key, value, dropout_p, is_causal, sliding_window,
                    sliding_window_size, attn_mask, scale, is_qk_quant,
                    is_pv_quant, num_threads);
        }
        return flash_attention_int8_mode_dispatch<Tag, bfloat16_t>(output,
                query, key, value, dropout_p, is_causal, sliding_window,
                sliding_window_size, attn_mask, scale, is_qk_quant, is_pv_quant,
                num_threads);
    };

    return use_avx512 ? run(simd::avx512_tag {}) : run(simd::scalar_tag {});
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
