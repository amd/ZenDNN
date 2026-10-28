/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

// Packed variable-length dynamic-INT8 flash-attention path.  Same scheme as
// lowoha_sdpa_flash_cpu_int8.cpp (Q/K per token, V per channel, U8 softmax
// tile), with sequences located by the cu_seqlens offsets.

#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_flash_cpu_common.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_int8_utils.hpp"

#include "common/logging.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include <atomic>

#include <omp.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

struct varlen_int8_scratch_buffers {
    std::vector<int8_t> q_s8;
    std::vector<int8_t> k_s8;
    std::vector<int8_t> v_s8;
    std::vector<float> q_scales;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
    std::vector<float> f32;
    std::vector<zendnnl::common::bfloat16_t> p_bf16;
    std::vector<uint8_t> p_u8;
    std::vector<varlen_tile> tiles;

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
        std::vector<varlen_tile>().swap(tiles);
    }
};

static thread_local varlen_int8_scratch_buffers g_varlen_int8_flash_scratch;

namespace {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,fma")
#pragma GCC optimize("no-tree-vectorize")
#endif

// Quantized Q/K/V are stored per head over all packed tokens:
//   q_s8 [num_heads][total_q][D],     q_scales [num_heads][total_q]
//   k_s8 [kv_num_heads][total_kv][D], k_scales [kv_num_heads][total_kv]
//   v_s8 [kv_num_heads][total_kv][D], v_scales [num_seqs][kv_num_heads][D]
// Q/K scales are per token, so quantizing a head across sequence boundaries
// matches quantizing each sequence alone. V scales are per channel, so they
// are computed over each sequence's own rows.
template <typename SimdTag, bool QkInt8, bool PvInt8>
status_t cpu_flash_attention_varlen_int8_dq(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, const int32_t *cu_seqlens_q,
        const int32_t *cu_seqlens_k, bool is_causal, bool sliding_window,
        int64_t sliding_window_size, std::optional<double> scale,
        int num_threads_hint) {
    SDPA_SA_CHECK(query.size_d == value.size_d && key.size_d == value.size_d,
            "Q/K/V head dim mismatch");
    SDPA_SA_CHECK(key.size_h > 0, "K/V head count must be > 0");
    SDPA_SA_CHECK(key.size_h == value.size_h, "K/V head count mismatch");
    SDPA_SA_CHECK(query.size_h % key.size_h == 0,
            "Q heads must be divisible by K/V heads");

    const int64_t num_seqs = query.size_b;
    const int64_t max_q = query.size_s;
    const int64_t max_kv = key.size_s;
    const int64_t num_head = query.size_h;
    const int64_t kv_num_head = key.size_h;
    const int64_t headSize = query.size_d;
    const int64_t repeat_factor = num_head / kv_num_head;
    if (num_seqs <= 0 || max_q <= 0) { return status_t::success; }
    const int64_t total_q = cu_seqlens_q[num_seqs];
    const int64_t total_kv = cu_seqlens_k[num_seqs];

    const accum_t scaling_factor = calculate_scale_value(scale, headSize);
    const bool use_window = sliding_window && sliding_window_size > 0;
    const bool banded = use_window || is_causal;
    const int64_t win_left = use_window ? sliding_window_size - 1 : 0;
    const int64_t win_right
            = (use_window && !is_causal) ? sliding_window_size - 1 : 0;

    const int64_t qSplitSize
            = std::min(varlen_q_split_size(num_seqs, max_q, use_window), max_q);
    const int64_t kvSplitSize
            = std::max<int64_t>(1, std::min<int64_t>(512, max_kv));
    auto &tiles = g_varlen_int8_flash_scratch.tiles;
    build_varlen_tiles(tiles, cu_seqlens_q, num_seqs, qSplitSize);
    const int64_t num_items = static_cast<int64_t>(tiles.size()) * num_head;
    const int num_thread
            = (num_threads_hint > 0) ? num_threads_hint : omp_get_max_threads();

    const int64_t qStrideM = query.stride_s;
    const int64_t qStrideH = query.stride_h;
    const int64_t kStrideN = key.stride_s;
    const int64_t kStrideH = key.stride_h;
    const int64_t vStrideN = value.stride_s;
    const int64_t vStrideH = value.stride_h;
    const int64_t oStrideM = output.stride_s;
    const int64_t oStrideH = output.stride_h;
    const int64_t oStrideD = output.stride_d;

    const auto *q_data = static_cast<const bfloat16_t *>(query.data);
    const auto *k_data = static_cast<const bfloat16_t *>(key.data);
    const auto *v_data = static_cast<const bfloat16_t *>(value.data);
    auto *out_data = static_cast<bfloat16_t *>(const_cast<void *>(output.data));

    // The grouped-quant API takes its extents as int.
    SDPA_SA_CHECK(total_q <= std::numeric_limits<int>::max()
                    && total_kv <= std::numeric_limits<int>::max()
                    && headSize <= std::numeric_limits<int>::max(),
            "INT8 grouped quantization dimensions exceed INT_MAX");

    auto &sc = g_varlen_int8_flash_scratch;
    const size_t D = static_cast<size_t>(headSize);
    if constexpr (QkInt8) {
        ensure_scratch_size(
                sc.q_s8, static_cast<size_t>(num_head * total_q) * D);
        ensure_scratch_size(
                sc.k_s8, static_cast<size_t>(kv_num_head * total_kv) * D);
        ensure_scratch_size(
                sc.q_scales, static_cast<size_t>(num_head * total_q));
        ensure_scratch_size(
                sc.k_scales, static_cast<size_t>(kv_num_head * total_kv));
    }
    if constexpr (PvInt8) {
        ensure_scratch_size(
                sc.v_s8, static_cast<size_t>(kv_num_head * total_kv) * D);
        ensure_scratch_size(
                sc.v_scales, static_cast<size_t>(num_seqs * kv_num_head) * D);
    }

    scoped_active_levels active_levels_guard(1);

    if constexpr (QkInt8) {
        std::vector<const void *> src;
        std::vector<int> rows, cols;
        std::vector<std::vector<int64_t>> strides;
        std::vector<void *> dst, scales;
        if (total_q > 0) {
            for (int64_t h = 0; h < num_head; ++h) {
                src.push_back(q_data + h * qStrideH);
                rows.push_back(static_cast<int>(total_q));
                cols.push_back(static_cast<int>(headSize));
                strides.push_back({qStrideM, 1});
                dst.push_back(
                        sc.q_s8.data() + static_cast<size_t>(h * total_q) * D);
                scales.push_back(
                        sc.q_scales.data() + static_cast<size_t>(h * total_q));
            }
        }
        if (total_kv > 0) {
            for (int64_t h = 0; h < kv_num_head; ++h) {
                src.push_back(k_data + h * kStrideH);
                rows.push_back(static_cast<int>(total_kv));
                cols.push_back(static_cast<int>(headSize));
                strides.push_back({kStrideN, 1});
                dst.push_back(
                        sc.k_s8.data() + static_cast<size_t>(h * total_kv) * D);
                scales.push_back(
                        sc.k_scales.data() + static_cast<size_t>(h * total_kv));
            }
        }
        if (!src.empty()
                && sdpa_group_quantize_bf16_s8_per_token(
                           src, rows, cols, strides, dst, scales, num_thread)
                        != status_t::success) {
            log_error("sdpa_flash_cpu_varlen: dynamic Q/K quantization failed");
            return status_t::failure;
        }
        if (scaling_factor != accum_t(1)) {
            // Carry the attention factor in the Q dequantization scale so
            // every score tile avoids a separate scaling pass.
            const size_t count = static_cast<size_t>(num_head * total_q);
            for (size_t index = 0; index < count; ++index) {
                sc.q_scales[index] *= scaling_factor;
            }
        }
    }

    if constexpr (PvInt8) {
        std::vector<const void *> src;
        std::vector<int> rows, cols;
        std::vector<std::vector<int64_t>> strides;
        std::vector<void *> dst, scales;
        for (int64_t b = 0; b < num_seqs; ++b) {
            const int64_t kv_start = cu_seqlens_k[b];
            const int64_t kv_len = cu_seqlens_k[b + 1] - kv_start;
            if (kv_len <= 0) { continue; }
            for (int64_t h = 0; h < kv_num_head; ++h) {
                src.push_back(v_data + kv_start * vStrideN + h * vStrideH);
                rows.push_back(static_cast<int>(kv_len));
                cols.push_back(static_cast<int>(headSize));
                strides.push_back({vStrideN, 1});
                dst.push_back(sc.v_s8.data()
                        + static_cast<size_t>(h * total_kv + kv_start) * D);
                scales.push_back(sc.v_scales.data()
                        + static_cast<size_t>(b * kv_num_head + h) * D);
            }
        }
        if (!src.empty()
                && sdpa_group_quantize_bf16_s8_per_channel(
                           src, rows, cols, strides, dst, scales, num_thread)
                        != status_t::success) {
            log_error("sdpa_flash_cpu_varlen: dynamic V quantization failed");
            return status_t::failure;
        }
    }

    const int64_t qk_tile_capacity = qSplitSize * kvSplitSize;
    const int64_t pv_tile_capacity = PvInt8 ? qSplitSize * headSize : 0;
    const int64_t size_per_thread = qk_tile_capacity + qSplitSize + qSplitSize
            + qSplitSize * headSize + pv_tile_capacity;
    ensure_scratch_size(
            sc.f32, static_cast<size_t>(num_thread * size_per_thread));
    if constexpr (PvInt8) {
        ensure_scratch_size(
                sc.p_u8, static_cast<size_t>(num_thread * qk_tile_capacity));
    } else {
        ensure_scratch_size(
                sc.p_bf16, static_cast<size_t>(num_thread * qk_tile_capacity));
    }

    constexpr float probability_scale = 1.0f / 255.0f;
    std::atomic<bool> execute_failed {false};
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_thread)
    for (int64_t z = 0; z < num_items; ++z) {
        if (execute_failed.load(std::memory_order_relaxed)) { continue; }

        const varlen_tile &tile = tiles[z / num_head];
        const int64_t i = tile.seq;
        const int64_t j = z % num_head;
        const int64_t m = tile.m;
        const int64_t qBlockSize = tile.rows;

        const int64_t q_start = cu_seqlens_q[i];
        const int64_t qSize = cu_seqlens_q[i + 1] - q_start;
        const int64_t kv_start = cu_seqlens_k[i];
        const int64_t kvSize = cu_seqlens_k[i + 1] - kv_start;
        const int64_t kv_j = j / repeat_factor;

        const int ompIdx = omp_get_thread_num();
        accum_t *qk_data = sc.f32.data() + ompIdx * size_per_thread;
        accum_t *qk_max_data = qk_data + qk_tile_capacity;
        accum_t *qk_sum_data = qk_max_data + qSplitSize;
        accum_t *dst_data = qk_sum_data + qSplitSize;
        accum_t *pv_data = PvInt8 ? dst_data + qSplitSize * headSize : nullptr;
        bfloat16_t *p_bf16_tile = PvInt8 ? nullptr
                                         : sc.p_bf16.data()
                        + static_cast<size_t>(ompIdx * qk_tile_capacity);
        uint8_t *p_u8_tile = PvInt8 ? sc.p_u8.data()
                        + static_cast<size_t>(ompIdx * qk_tile_capacity)
                                    : nullptr;

        const size_t q_row = static_cast<size_t>(j * total_q + q_start + m);
        const size_t kv_row = static_cast<size_t>(kv_j * total_kv + kv_start);
        const int8_t *q_tile = QkInt8 ? sc.q_s8.data() + q_row * D : nullptr;
        const float *q_tile_scales
                = QkInt8 ? sc.q_scales.data() + q_row : nullptr;
        const int8_t *k_seq_s8 = QkInt8 ? sc.k_s8.data() + kv_row * D : nullptr;
        const float *k_seq_scales
                = QkInt8 ? sc.k_scales.data() + kv_row : nullptr;
        const bfloat16_t *q_bf16_tile = QkInt8
                ? nullptr
                : q_data + (q_start + m) * qStrideM + j * qStrideH;
        const bfloat16_t *k_bf16_seq = QkInt8
                ? nullptr
                : k_data + kv_start * kStrideN + kv_j * kStrideH;
        const int8_t *v_seq_s8 = PvInt8 ? sc.v_s8.data() + kv_row * D : nullptr;
        const float *v_seq_scales = PvInt8 ? sc.v_scales.data()
                        + static_cast<size_t>(i * kv_num_head + kv_j) * D
                                           : nullptr;
        const bfloat16_t *v_bf16_seq
                = v_data + kv_start * vStrideN + kv_j * vStrideH;
        bfloat16_t *o_tile = out_data + (q_start + m) * oStrideM + j * oStrideH;

        fill_stub_f32<SimdTag>(qk_max_data,
                -std::numeric_limits<accum_t>::infinity(), qBlockSize);
        fill_stub_f32<SimdTag>(
                qk_sum_data, static_cast<accum_t>(0), qBlockSize);

        // Queries are aligned to the end of their keys (see the FP kernel).
        const int64_t q_pos = m + kvSize - qSize;
        int64_t num_keys = kvSize;
        int64_t n_start = 0;
        if (banded) {
            num_keys = std::min(kvSize, q_pos + qBlockSize + win_right);
        }
        if (use_window) {
            n_start = std::max<int64_t>(0, q_pos - win_left) / kvSplitSize
                    * kvSplitSize;
        }

        bool have_kv = false;
        for (int64_t n = n_start; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            int64_t n_gemm = n;
            if (banded) {
                const int64_t band_lo
                        = use_window ? std::max(n, q_pos - win_left) : n;
                const int64_t band_hi = std::min(
                        n + kvBlockSize, q_pos + qBlockSize + win_right);
                if (band_lo >= band_hi) { continue; }
                n_gemm = band_lo;
                kvBlockSize = band_hi - band_lo;
            }
            status_t st = status_t::success;
            if constexpr (QkInt8) {
                st = sdpa_qk_int8_matmul(q_tile,
                        k_seq_s8 + static_cast<size_t>(n_gemm) * D, qk_data,
                        static_cast<int>(qBlockSize),
                        static_cast<int>(kvBlockSize),
                        static_cast<int>(headSize), q_tile_scales,
                        k_seq_scales + n_gemm);
                if (st != status_t::success) {
                    execute_failed.store(true, std::memory_order_relaxed);
                    break;
                }
            } else {
                // No Q scale to carry the attention factor, so fold it into
                // the GEMM alpha.
                zendnn_gemm<bfloat16_t>(qBlockSize, kvBlockSize, headSize,
                        scaling_factor, q_bf16_tile, qStrideM,
                        k_bf16_seq + n_gemm * kStrideN, kStrideN, 0.0f, qk_data,
                        kvBlockSize, false, true);
            }

            if (banded) {
                constexpr accum_t neg_inf
                        = -std::numeric_limits<accum_t>::infinity();
                for (int64_t row = 0; row < qBlockSize; ++row) {
                    const int64_t pos = q_pos + row;
                    const int64_t lo = use_window
                            ? std::max<int64_t>(0, pos - win_left - n_gemm)
                            : 0;
                    const int64_t hi = std::min(
                            kvBlockSize - 1, pos + win_right - n_gemm);
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
            }

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
                tmp_max = row_max<SimdTag>(
                        qk_data + row * kvBlockSize, kvBlockSize);
                tmp_max = std::max(qk_max_data[row], tmp_max);
                accum_t *qk_row = qk_data + row * kvBlockSize;
                uint8_t *p_u8_row
                        = PvInt8 ? p_u8_tile + row * kvBlockSize : nullptr;
                if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
                    if constexpr (PvInt8) {
                        std::fill_n(p_u8_row, kvBlockSize, uint8_t(0));
                    } else {
                        std::fill_n(qk_row, kvBlockSize, accum_t(0));
                    }
                    continue;
                }
                tmp_sum = tmp_max;
                if constexpr (PvInt8) {
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

            if constexpr (PvInt8) {
                accum_t *pv_output = have_kv ? pv_data : dst_data;
                st = sdpa_pv_int8_matmul(p_u8_tile,
                        v_seq_s8 + static_cast<size_t>(n_gemm) * D, pv_output,
                        static_cast<int>(qBlockSize),
                        static_cast<int>(headSize),
                        static_cast<int>(kvBlockSize), &probability_scale,
                        v_seq_scales);
                if (st != status_t::success) {
                    execute_failed.store(true, std::memory_order_relaxed);
                    break;
                }
                if (have_kv) {
                    add_f32_inplace<SimdTag>(
                            dst_data, pv_data, qBlockSize * headSize);
                }
            } else {
                vec_f32_scaled_bf16_store<SimdTag>(p_bf16_tile, qk_data,
                        qBlockSize * kvBlockSize, accum_t(1));
                zendnn_gemm<bfloat16_t>(qBlockSize, headSize, kvBlockSize, 1.0f,
                        p_bf16_tile, kvBlockSize,
                        v_bf16_seq + n_gemm * vStrideN, vStrideN,
                        have_kv ? 1.0f : 0.0f, dst_data, headSize, false,
                        false);
            }
            have_kv = true;
        }

        if (execute_failed.load(std::memory_order_relaxed)) { continue; }
        if (!have_kv) {
            fill_stub_f32<SimdTag>(
                    dst_data, static_cast<accum_t>(0), qBlockSize * headSize);
        }
        for (int64_t row = 0; row < qBlockSize; ++row) {
            if (qk_sum_data[row] == 0) { qk_sum_data[row] = 1; }
            const accum_t sum_reciprocal = 1 / qk_sum_data[row];
            write_scaled_output_row<SimdTag, bfloat16_t>(
                    o_tile + row * oStrideM, oStrideD,
                    dst_data + row * headSize, headSize, sum_reciprocal);
        }
    }

    if (execute_failed.load(std::memory_order_relaxed)) {
        log_error("sdpa_flash_cpu_varlen: dynamic INT8 matmul failed");
        return status_t::failure;
    }
    return status_t::success;
}

template <typename SimdTag>
status_t flash_attention_varlen_int8_mode_dispatch(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, const int32_t *cu_seqlens_q,
        const int32_t *cu_seqlens_k, bool is_causal, bool sliding_window,
        int64_t sliding_window_size, std::optional<double> scale,
        bool is_qk_quant, bool is_pv_quant, int num_threads) {
    if (is_qk_quant && is_pv_quant) {
        return cpu_flash_attention_varlen_int8_dq<SimdTag, true, true>(output,
                query, key, value, cu_seqlens_q, cu_seqlens_k, is_causal,
                sliding_window, sliding_window_size, scale, num_threads);
    }
    if (is_qk_quant) {
        return cpu_flash_attention_varlen_int8_dq<SimdTag, true, false>(output,
                query, key, value, cu_seqlens_q, cu_seqlens_k, is_causal,
                sliding_window, sliding_window_size, scale, num_threads);
    }
    return cpu_flash_attention_varlen_int8_dq<SimdTag, false, true>(output,
            query, key, value, cu_seqlens_q, cu_seqlens_k, is_causal,
            sliding_window, sliding_window_size, scale, num_threads);
}

} // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void sdpa_flash_cpu_free_varlen_int8_scratch() {
    g_varlen_int8_flash_scratch.release();
}

status_t sdpa_flash_cpu_varlen_run_int8(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, const int32_t *cu_seqlens_q,
        const int32_t *cu_seqlens_k, bool is_causal, bool sliding_window,
        int64_t sliding_window_size, std::optional<double> scale,
        data_type_t qkv_dt, bool is_qk_quant, bool is_pv_quant, bool use_avx512,
        int num_threads) {
    if (qkv_dt != data_type_t::bf16) {
        log_error(
                "sdpa_flash_cpu_varlen: dynamic INT8 path requires BF16 "
                "Q/K/V");
        return status_t::unimplemented;
    }
    if (!is_qk_quant && !is_pv_quant) {
        log_error(
                "sdpa_flash_cpu_varlen: INT8 path needs is_qk_quant or "
                "is_pv_quant");
        return status_t::failure;
    }
    return use_avx512
            ? flash_attention_varlen_int8_mode_dispatch<simd::avx512_tag>(
                      output, query, key, value, cu_seqlens_q, cu_seqlens_k,
                      is_causal, sliding_window, sliding_window_size, scale,
                      is_qk_quant, is_pv_quant, num_threads)
            : flash_attention_varlen_int8_mode_dispatch<simd::scalar_tag>(
                      output, query, key, value, cu_seqlens_q, cu_seqlens_k,
                      is_causal, sliding_window, sliding_window_size, scale,
                      is_qk_quant, is_pv_quant, num_threads);
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
