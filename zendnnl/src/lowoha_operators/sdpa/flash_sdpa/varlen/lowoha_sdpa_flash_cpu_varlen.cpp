/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

// Packed variable-length FP32 / BF16 / FP16 flash-attention path.  Sequences
// are stored back to back as [total_tokens, heads, head_dim] and located by
// the cu_seqlens offsets; the dense BHSD path lives in
// lowoha_sdpa_flash_cpu.cpp and the shared helpers in
// lowoha_sdpa_flash_cpu_common.hpp.

#include "lowoha_operators/sdpa/flash_sdpa/lowoha_sdpa_flash_cpu_common.hpp"

#include "common/logging.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include <omp.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace zendnnl {
namespace lowoha {
namespace sdpa {

struct varlen_scratch_buffers {
    std::vector<accum_t> f32;
    std::vector<uint16_t> reduced;
    std::vector<varlen_tile> tiles;

    void release() {
        std::vector<accum_t>().swap(f32);
        std::vector<uint16_t>().swap(reduced);
        std::vector<varlen_tile>().swap(tiles);
    }
};

static thread_local varlen_scratch_buffers g_varlen_flash_scratch;

namespace {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,fma")
#pragma GCC optimize("no-tree-vectorize")
#endif

template <typename SimdTag, typename scalar_t>
void cpu_flash_attention_varlen(const sdpa_flash_cpu_tensor_view &output,
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

    constexpr bool is_reduced_type = is_reduced_fp_v<scalar_t>;
    const int64_t num_seqs = query.size_b;
    const int64_t max_q = query.size_s;
    const int64_t max_kv = key.size_s;
    const int64_t num_head = query.size_h;
    const int64_t headSize = query.size_d;
    const int64_t repeat_factor = num_head / key.size_h;
    if (num_seqs <= 0 || max_q <= 0) { return; }

    const accum_t scaling_factor = calculate_scale_value(scale, headSize);

    // Each query row sees keys in [pos - win_left, pos + win_right] (window)
    // and none after pos (causal), where pos is its key position.
    const bool use_window = sliding_window && sliding_window_size > 0;
    const bool banded = use_window || is_causal;
    const int64_t win_left = use_window ? sliding_window_size - 1 : 0;
    const int64_t win_right
            = (use_window && !is_causal) ? sliding_window_size - 1 : 0;

    const int64_t num_thread
            = (num_threads_hint > 0) ? num_threads_hint : omp_get_max_threads();
    const auto tasks_for = [&](int64_t split) {
        return num_head * varlen_tile_count(cu_seqlens_q, num_seqs, split);
    };
    const int64_t qSplitSize = balanced_q_split_size(
            std::min(varlen_q_split_size(num_seqs, max_q, use_window), max_q),
            num_thread, tasks_for, varlen_min_gain_percent);
    const int64_t kvSplitSize
            = std::max<int64_t>(1, std::min<int64_t>(512, max_kv));
    auto &tiles = g_varlen_flash_scratch.tiles;
    build_varlen_tiles(tiles, cu_seqlens_q, num_seqs, qSplitSize);
    const int64_t num_items = static_cast<int64_t>(tiles.size()) * num_head;

    const int64_t qStrideM = query.stride_s;
    const int64_t qStrideH = query.stride_h;
    const int64_t kStrideN = key.stride_s;
    const int64_t kStrideH = key.stride_h;
    const int64_t vStrideN = value.stride_s;
    const int64_t vStrideH = value.stride_h;
    const int64_t oStrideM = output.stride_s;
    const int64_t oStrideH = output.stride_h;
    const int64_t oStrideD = output.stride_d;

    const int64_t size_per_thread = qSplitSize * kvSplitSize + qSplitSize
            + qSplitSize + qSplitSize * headSize;
    const int64_t reduced_per_thread
            = is_reduced_type ? qSplitSize * kvSplitSize : 0;
    ensure_scratch_size(g_varlen_flash_scratch.f32,
            static_cast<size_t>(num_thread * size_per_thread));
    if constexpr (is_reduced_type) {
        ensure_scratch_size(g_varlen_flash_scratch.reduced,
                static_cast<size_t>(num_thread * reduced_per_thread));
    }
    accum_t *buf_data = g_varlen_flash_scratch.f32.data();
    scalar_t *buf_reduced_data = is_reduced_type
            ? reinterpret_cast<scalar_t *>(
                      g_varlen_flash_scratch.reduced.data())
            : nullptr;

    const auto *q_data = static_cast<const scalar_t *>(query.data);
    const auto *k_data = static_cast<const scalar_t *>(key.data);
    const auto *v_data = static_cast<const scalar_t *>(value.data);
    auto *out_data = static_cast<scalar_t *>(const_cast<void *>(output.data));

    scoped_active_levels active_levels_guard(1);

    // Tile costs differ across sequences, so hand work out dynamically. The
    // heads of one tile are adjacent, so heads sharing a K/V head run close
    // together.
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_thread)
    for (int64_t z = 0; z < num_items; ++z) {
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
        const scalar_t *q_tile
                = q_data + (q_start + m) * qStrideM + j * qStrideH;
        const scalar_t *k_seq = k_data + kv_start * kStrideN + kv_j * kStrideH;
        const scalar_t *v_seq = v_data + kv_start * vStrideN + kv_j * vStrideH;
        scalar_t *o_tile = out_data + (q_start + m) * oStrideM + j * oStrideH;

        const int ompIdx = omp_get_thread_num();
        accum_t *qk_data = buf_data + ompIdx * size_per_thread;
        accum_t *qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t *qk_sum_data = qk_max_data + qSplitSize;
        accum_t *dst_data = qk_sum_data + qSplitSize;
        scalar_t *qk_reduced_data = is_reduced_type
                ? buf_reduced_data + ompIdx * reduced_per_thread
                : nullptr;

        fill_stub_f32<SimdTag>(qk_max_data,
                -std::numeric_limits<accum_t>::infinity(), qBlockSize);
        fill_stub_f32<SimdTag>(
                qk_sum_data, static_cast<accum_t>(0), qBlockSize);

        // Queries are aligned to the end of their keys: query row r sits at
        // key position r + (kvSize - qSize), so a prefill chunk after a cached
        // prefix sees the whole prefix. Equal lengths give position r.
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
            zendnn_gemm<scalar_t>(qBlockSize, kvBlockSize, headSize, 1.0f,
                    q_tile, qStrideM, k_seq + n_gemm * kStrideN, kStrideN, 0.0f,
                    qk_data, kvBlockSize, false, true);

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
                mul_reduce_max_fusion<SimdTag>(qk_data + row * kvBlockSize,
                        scaling_factor, static_cast<int>(kvBlockSize),
                        qk_data + row * kvBlockSize, tmp_max);
                tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row]
                                                     : tmp_max;
                scalar_t *qk_row
                        = conditional_data_ptr(qk_data, qk_reduced_data)
                        + row * kvBlockSize;
                // A row whose band misses this whole tile has max=-inf;
                // exponentiating would produce NaN, so give it zero weight.
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
            zendnn_gemm<scalar_t>(qBlockSize, headSize, kvBlockSize, 1.0f,
                    conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
                    v_seq + n_gemm * vStrideN, vStrideN, have_kv ? 1.0f : 0.0f,
                    dst_data, headSize, false, false);
            have_kv = true;
        }

        // No key/value tile ran (empty key sequence, a window that misses
        // every key, or causal rows of a chunk longer than its keys), so the
        // P*V GEMM never initialised the accumulator.
        if (!have_kv) {
            fill_stub_f32<SimdTag>(
                    dst_data, static_cast<accum_t>(0), qBlockSize * headSize);
        }

        for (int64_t row = 0; row < qBlockSize; ++row) {
            if (qk_sum_data[row] == 0) { qk_sum_data[row] = 1; }
            const accum_t sum_reciprocal = 1 / qk_sum_data[row];
            write_scaled_output_row<SimdTag, scalar_t>(o_tile + row * oStrideM,
                    oStrideD, dst_data + row * headSize, headSize,
                    sum_reciprocal);
        }
    }
}

} // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void sdpa_flash_cpu_free_varlen_scratch() {
    g_varlen_flash_scratch.release();
}

status_t sdpa_flash_cpu_varlen_run_internal(
        const sdpa_flash_cpu_tensor_view &output,
        const sdpa_flash_cpu_tensor_view &query,
        const sdpa_flash_cpu_tensor_view &key,
        const sdpa_flash_cpu_tensor_view &value, const int32_t *cu_seqlens_q,
        const int32_t *cu_seqlens_k, bool is_causal, bool sliding_window,
        int64_t sliding_window_size, const double *scale_opt,
        data_type_t qkv_dt, bool is_qk_quant, bool is_pv_quant,
        int num_threads) {
    try {
        std::optional<double> scale;
        if (scale_opt != nullptr) { scale = *scale_opt; }

        const auto &platform = zendnnl::common::zendnnl_platform_info();
        if (qkv_dt == data_type_t::f16 && !platform.get_avx512_f16_status()) {
            log_error("sdpa_flash_cpu_varlen: FP16 requires AVX512-FP16");
            return status_t::isa_unsupported;
        }
        const bool any_int8_quant = is_qk_quant || is_pv_quant;
        if (any_int8_quant
                && (!platform.get_avx512f_status()
                        || !platform.get_avx512_bw_vl_status()
                        || !platform.get_avx512_vnni_status())) {
            log_error(
                    "sdpa_flash_cpu_varlen: dynamic INT8 path requires "
                    "AVX512-F, AVX512-BW, AVX512-VL, and AVX512-VNNI");
            return status_t::isa_unsupported;
        }

        const bool use_avx512 = platform.get_avx512f_status();
        apilog_info(use_avx512 ? "sdpa_flash_cpu_varlen: using AVX-512 SIMD"
                               : "sdpa_flash_cpu_varlen: using scalar SIMD");

        if (any_int8_quant) {
            return sdpa_flash_cpu_varlen_run_int8(output, query, key, value,
                    cu_seqlens_q, cu_seqlens_k, is_causal, sliding_window,
                    sliding_window_size, scale, qkv_dt, is_qk_quant,
                    is_pv_quant, use_avx512, num_threads);
        }

        auto run = [&](auto simd_tag) -> status_t {
            using Tag = decltype(simd_tag);
            if (qkv_dt == data_type_t::f32) {
                cpu_flash_attention_varlen<Tag, float>(output, query, key,
                        value, cu_seqlens_q, cu_seqlens_k, is_causal,
                        sliding_window, sliding_window_size, scale,
                        num_threads);
            } else if (qkv_dt == data_type_t::bf16) {
                cpu_flash_attention_varlen<Tag, bfloat16_t>(output, query, key,
                        value, cu_seqlens_q, cu_seqlens_k, is_causal,
                        sliding_window, sliding_window_size, scale,
                        num_threads);
            } else if (qkv_dt == data_type_t::f16) {
                cpu_flash_attention_varlen<Tag, float16_t>(output, query, key,
                        value, cu_seqlens_q, cu_seqlens_k, is_causal,
                        sliding_window, sliding_window_size, scale,
                        num_threads);
            } else {
                log_error("sdpa_flash_cpu_varlen: unsupported Q/K/V dtype");
                throw std::invalid_argument("unsupported Q/K/V dtype");
            }
            return status_t::success;
        };

        return use_avx512 ? run(simd::avx512_tag {}) : run(simd::scalar_tag {});
    } catch (const std::invalid_argument &e) {
        log_error("sdpa_flash_cpu_varlen: invalid argument: ", e.what());
        return status_t::failure;
    } catch (const std::exception &e) {
        log_error("sdpa_flash_cpu_varlen: execution failed: ", e.what());
        return status_t::failure;
    }
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
