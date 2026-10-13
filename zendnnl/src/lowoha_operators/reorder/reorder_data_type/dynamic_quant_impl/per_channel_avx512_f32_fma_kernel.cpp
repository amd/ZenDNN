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
 ******************************************************************************/

#include "common/bfloat16.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace reorder {

namespace {

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 load_bf16x16_as_f32(const uint16_t *src) {
    const __m256i bf16
            = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 finite_abs(
        __m512 value, __m512i abs_mask, __m512 infinity, __mmask16 &mask) {
    const __m512 abs_value = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(value), abs_mask));
    mask = _mm512_cmp_ps_mask(abs_value, infinity, _CMP_LT_OQ);
    return abs_value;
}

// Scale and quantize both divide instead of multiplying by a reciprocal: the
// rounded reciprocal differs from the scalar tail and the reference by one ULP,
// which flips s8 levels at values exactly half-way between two levels.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
inline __m128i quantize_bf16x16(__m512 value, __m512 scale, __m512i lower_bound,
        __m512i upper_bound, __m512i abs_mask, __m512 infinity) {
    __mmask16 finite;
    finite_abs(value, abs_mask, infinity, finite);
    value = _mm512_maskz_mov_ps(finite, value);
    __m512i quantized = _mm512_cvtps_epi32(_mm512_div_ps(value, scale));
    quantized = _mm512_max_epi32(
            lower_bound, _mm512_min_epi32(upper_bound, quantized));
    return _mm512_cvtepi32_epi8(quantized);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void quantize_bf16_s8_per_channel_16(const uint16_t *src, int64_t src_lda,
        int8_t *dst, int64_t dst_lda, float *scales, int64_t rows,
        int64_t column) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512 infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
    __m512 absmax = _mm512_setzero_ps();

    for (int64_t row = 0; row < rows; ++row) {
        const __m512 value = load_bf16x16_as_f32(src + row * src_lda + column);
        __mmask16 finite;
        const __m512 abs_value = finite_abs(value, abs_mask, infinity, finite);
        absmax = _mm512_mask_max_ps(absmax, finite, absmax, abs_value);
    }

    const __m512 scale
            = _mm512_max_ps(_mm512_div_ps(absmax, _mm512_set1_ps(127.0f)),
                    _mm512_set1_ps(1.0e-10f));
    _mm512_storeu_ps(scales + column, scale);
    const __m512i lower_bound = _mm512_set1_epi32(-128);
    const __m512i upper_bound = _mm512_set1_epi32(127);

    for (int64_t row = 0; row < rows; ++row) {
        const __m512 value = load_bf16x16_as_f32(src + row * src_lda + column);
        const __m128i quantized = quantize_bf16x16(
                value, scale, lower_bound, upper_bound, abs_mask, infinity);
        _mm_storeu_si128(
                reinterpret_cast<__m128i *>(dst + row * dst_lda + column),
                quantized);
    }
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void quantize_bf16_s8_per_channel_64(const uint16_t *src, int64_t src_lda,
        int8_t *dst, int64_t dst_lda, float *scales, int64_t rows,
        int64_t column) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512 infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
    __m512 absmax0 = _mm512_setzero_ps();
    __m512 absmax1 = _mm512_setzero_ps();
    __m512 absmax2 = _mm512_setzero_ps();
    __m512 absmax3 = _mm512_setzero_ps();

    for (int64_t row = 0; row < rows; ++row) {
        const auto *row_src = src + row * src_lda + column;
        const __m512 value0 = load_bf16x16_as_f32(row_src);
        const __m512 value1 = load_bf16x16_as_f32(row_src + 16);
        const __m512 value2 = load_bf16x16_as_f32(row_src + 32);
        const __m512 value3 = load_bf16x16_as_f32(row_src + 48);
        __mmask16 finite0, finite1, finite2, finite3;
        const __m512 abs_value0
                = finite_abs(value0, abs_mask, infinity, finite0);
        const __m512 abs_value1
                = finite_abs(value1, abs_mask, infinity, finite1);
        const __m512 abs_value2
                = finite_abs(value2, abs_mask, infinity, finite2);
        const __m512 abs_value3
                = finite_abs(value3, abs_mask, infinity, finite3);
        absmax0 = _mm512_mask_max_ps(absmax0, finite0, absmax0, abs_value0);
        absmax1 = _mm512_mask_max_ps(absmax1, finite1, absmax1, abs_value1);
        absmax2 = _mm512_mask_max_ps(absmax2, finite2, absmax2, abs_value2);
        absmax3 = _mm512_mask_max_ps(absmax3, finite3, absmax3, abs_value3);
    }

    const __m512 quant_max = _mm512_set1_ps(127.0f);
    const __m512 minimum_scale = _mm512_set1_ps(1.0e-10f);
    const __m512 scale0
            = _mm512_max_ps(_mm512_div_ps(absmax0, quant_max), minimum_scale);
    const __m512 scale1
            = _mm512_max_ps(_mm512_div_ps(absmax1, quant_max), minimum_scale);
    const __m512 scale2
            = _mm512_max_ps(_mm512_div_ps(absmax2, quant_max), minimum_scale);
    const __m512 scale3
            = _mm512_max_ps(_mm512_div_ps(absmax3, quant_max), minimum_scale);
    _mm512_storeu_ps(scales + column, scale0);
    _mm512_storeu_ps(scales + column + 16, scale1);
    _mm512_storeu_ps(scales + column + 32, scale2);
    _mm512_storeu_ps(scales + column + 48, scale3);

    const __m512i lower_bound = _mm512_set1_epi32(-128);
    const __m512i upper_bound = _mm512_set1_epi32(127);

    for (int64_t row = 0; row < rows; ++row) {
        const auto *row_src = src + row * src_lda + column;
        const __m128i quantized0
                = quantize_bf16x16(load_bf16x16_as_f32(row_src), scale0,
                        lower_bound, upper_bound, abs_mask, infinity);
        const __m128i quantized1
                = quantize_bf16x16(load_bf16x16_as_f32(row_src + 16), scale1,
                        lower_bound, upper_bound, abs_mask, infinity);
        const __m128i quantized2
                = quantize_bf16x16(load_bf16x16_as_f32(row_src + 32), scale2,
                        lower_bound, upper_bound, abs_mask, infinity);
        const __m128i quantized3
                = quantize_bf16x16(load_bf16x16_as_f32(row_src + 48), scale3,
                        lower_bound, upper_bound, abs_mask, infinity);
        const __m256i lower = _mm256_set_m128i(quantized1, quantized0);
        const __m256i upper = _mm256_set_m128i(quantized3, quantized2);
        const __m512i packed
                = _mm512_inserti64x4(_mm512_castsi256_si512(lower), upper, 1);
        _mm512_storeu_si512(dst + row * dst_lda + column, packed);
    }
}

void quantize_bf16_s8_per_channel_scalar(const uint16_t *src, int64_t src_lda,
        int8_t *dst, int64_t dst_lda, float *scales, int64_t rows,
        int64_t column_begin, int64_t column_end) {
    for (int64_t column = column_begin; column < column_end; ++column) {
        float absmax = 0.0f;
        for (int64_t row = 0; row < rows; ++row) {
            const float value = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(src[row * src_lda + column]));
            if (std::isfinite(value)) {
                absmax = std::max(absmax, std::fabs(value));
            }
        }

        const float scale = std::max(absmax / 127.0f, 1.0e-10f);
        scales[column] = scale;
        for (int64_t row = 0; row < rows; ++row) {
            const float value = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(src[row * src_lda + column]));
            int32_t quantized = 0;
            if (std::isfinite(value)) {
                quantized = static_cast<int32_t>(std::nearbyint(value / scale));
                quantized = std::max(-128, std::min(127, quantized));
            }
            dst[row * dst_lda + column] = static_cast<int8_t>(quantized);
        }
    }
}

// absmax[c] = max(absmax[c], |src[r][c]|) over finite values in `rows`;
// the caller zero-initialises absmax.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void accumulate_bf16_column_absmax(const uint16_t *src, int64_t src_lda,
        int64_t rows, int64_t columns, float *absmax) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512 infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
    const int64_t vector_columns = columns / 16 * 16;
    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *row_src = src + row * src_lda;
        for (int64_t column = 0; column < vector_columns; column += 16) {
            __mmask16 finite;
            const __m512 abs_value
                    = finite_abs(load_bf16x16_as_f32(row_src + column),
                            abs_mask, infinity, finite);
            const __m512 current = _mm512_loadu_ps(absmax + column);
            _mm512_storeu_ps(absmax + column,
                    _mm512_mask_max_ps(current, finite, current, abs_value));
        }
        for (int64_t column = vector_columns; column < columns; ++column) {
            const float value = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(row_src[column]));
            if (std::isfinite(value)) {
                absmax[column] = std::max(absmax[column], std::fabs(value));
            }
        }
    }
}

// Reduces `parts` absmax arrays of `columns` floats, laid out back to back,
// into per-channel scales.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void absmax_parts_to_scales(const float *absmax_parts, int64_t parts,
        int64_t columns, float *scales) {
    const __m512 quant_max = _mm512_set1_ps(127.0f);
    const __m512 minimum_scale = _mm512_set1_ps(1.0e-10f);
    const int64_t vector_columns = columns / 16 * 16;
    for (int64_t column = 0; column < vector_columns; column += 16) {
        __m512 absmax = _mm512_loadu_ps(absmax_parts + column);
        for (int64_t part = 1; part < parts; ++part) {
            absmax = _mm512_max_ps(absmax,
                    _mm512_loadu_ps(absmax_parts + part * columns + column));
        }
        _mm512_storeu_ps(scales + column,
                _mm512_max_ps(_mm512_div_ps(absmax, quant_max), minimum_scale));
    }
    for (int64_t column = vector_columns; column < columns; ++column) {
        float absmax = absmax_parts[column];
        for (int64_t part = 1; part < parts; ++part) {
            absmax = std::max(absmax, absmax_parts[part * columns + column]);
        }
        scales[column] = std::max(absmax / 127.0f, 1.0e-10f);
    }
}

// Quantizes `rows` whole rows with precomputed per-channel scales.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void quantize_bf16_s8_rows_with_scales(const uint16_t *src, int64_t src_lda,
        int8_t *dst, int64_t dst_lda, const float *scales, int64_t rows,
        int64_t columns) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512 infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
    const __m512i lower_bound = _mm512_set1_epi32(-128);
    const __m512i upper_bound = _mm512_set1_epi32(127);
    const int64_t vector_columns = columns / 16 * 16;
    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *row_src = src + row * src_lda;
        int8_t *row_dst = dst + row * dst_lda;
        for (int64_t column = 0; column < vector_columns; column += 16) {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + column),
                    quantize_bf16x16(load_bf16x16_as_f32(row_src + column),
                            _mm512_loadu_ps(scales + column), lower_bound,
                            upper_bound, abs_mask, infinity));
        }
        for (int64_t column = vector_columns; column < columns; ++column) {
            const float value = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(row_src[column]));
            int32_t quantized = 0;
            if (std::isfinite(value)) {
                quantized = static_cast<int32_t>(
                        std::nearbyint(value / scales[column]));
                quantized = std::max(-128, std::min(127, quantized));
            }
            row_dst[column] = static_cast<int8_t>(quantized);
        }
    }
}

int requested_team_size(int num_threads) {
    return std::max(1, num_threads > 0 ? num_threads : omp_get_max_threads());
}

// Used when any active matrix has K not a multiple of 64, or when there are
// fewer 64-column blocks than threads (few, narrow matrices such as a single
// SDPA V head).  Each matrix's rows are cut into
// chunks: every chunk accumulates a partial column absmax, the partials of a
// matrix are reduced into its scales, then every chunk quantizes its rows.
// Chunks write whole output rows, so threads never share an output cache line
// the way 16-column blocks do.
void dynamic_per_channel_group_quant_rows(const std::vector<const void *> &src,
        const std::vector<int> &M, const std::vector<int> &K,
        const std::vector<int> &lda, const std::vector<void *> &dst,
        const std::vector<int> &dst_lda, const std::vector<float *> &scales,
        int thread_count) {
    const size_t num_ops = M.size();
    int64_t active_ops = 0;
    for (size_t op = 0; op < num_ops; ++op) {
        if (M[op] > 0) ++active_ops;
    }
    if (active_ops == 0) return;

    // At least one chunk per thread, and small enough that a chunk is still
    // cached when it is re-read for quantization.
    constexpr int64_t chunk_bytes = 256 * 1024;
    const int64_t min_chunks = (thread_count + active_ops - 1) / active_ops;
    std::vector<int64_t> chunk_prefix(num_ops), chunk_rows(num_ops),
            absmax_offset(num_ops);
    int64_t total_chunks = 0;
    int64_t total_absmax = 0;
    for (size_t op = 0; op < num_ops; ++op) {
        int64_t chunks = 0;
        if (M[op] > 0) {
            const int64_t max_rows = std::max<int64_t>(1,
                    chunk_bytes / (int64_t(K[op]) * int64_t(sizeof(uint16_t))));
            chunks = std::min<int64_t>(M[op],
                    std::max(min_chunks, (M[op] + max_rows - 1) / max_rows));
            chunk_rows[op] = (M[op] + chunks - 1) / chunks;
        }
        absmax_offset[op] = total_absmax;
        total_absmax += chunks * K[op];
        total_chunks += chunks;
        chunk_prefix[op] = total_chunks;
    }
    std::vector<float> absmax(static_cast<size_t>(total_absmax), 0.0f);

    struct chunk_t {
        size_t op;
        int64_t index, row_begin, rows;
    };
    // chunk_prefix holds running chunk counts, so the first entry above
    // task identifies the matrix that owns this chunk.
    auto chunk_at = [&](int64_t task) {
        chunk_t c;
        c.op = static_cast<size_t>(
                std::upper_bound(chunk_prefix.begin(), chunk_prefix.end(), task)
                - chunk_prefix.begin());
        c.index = task - (c.op == 0 ? 0 : chunk_prefix[c.op - 1]);
        c.row_begin = std::min<int64_t>(M[c.op], c.index * chunk_rows[c.op]);
        c.rows = std::min<int64_t>(M[c.op], c.row_begin + chunk_rows[c.op])
                - c.row_begin;
        return c;
    };

    const int team_size
            = static_cast<int>(std::min<int64_t>(thread_count, total_chunks));
#pragma omp parallel num_threads(team_size)
    {
#pragma omp for schedule(static)
        for (int64_t task = 0; task < total_chunks; ++task) {
            const chunk_t c = chunk_at(task);
            accumulate_bf16_column_absmax(
                    static_cast<const uint16_t *>(src[c.op])
                            + c.row_begin * lda[c.op],
                    lda[c.op], c.rows, K[c.op],
                    absmax.data() + absmax_offset[c.op] + c.index * K[c.op]);
        }

#pragma omp for schedule(static)
        for (int64_t op = 0; op < static_cast<int64_t>(num_ops); ++op) {
            if (M[op] <= 0) continue;
            const int64_t parts
                    = chunk_prefix[op] - (op == 0 ? 0 : chunk_prefix[op - 1]);
            absmax_parts_to_scales(absmax.data() + absmax_offset[op], parts,
                    K[op], scales[op]);
        }

#pragma omp for schedule(static)
        for (int64_t task = 0; task < total_chunks; ++task) {
            const chunk_t c = chunk_at(task);
            quantize_bf16_s8_rows_with_scales(
                    static_cast<const uint16_t *>(src[c.op])
                            + c.row_begin * lda[c.op],
                    lda[c.op],
                    static_cast<int8_t *>(dst[c.op])
                            + c.row_begin * dst_lda[c.op],
                    dst_lda[c.op], scales[c.op], c.rows, K[c.op]);
        }
    }
}

} // namespace

void dynamic_per_channel_quant_bf16_s8_native(const uint16_t *src,
        int64_t src_lda, int8_t *dst, int64_t dst_lda, float *scales,
        int64_t rows, int64_t column_begin, int64_t column_end) {
    int64_t column = column_begin;
    for (; column + 64 <= column_end; column += 64) {
        quantize_bf16_s8_per_channel_64(
                src, src_lda, dst, dst_lda, scales, rows, column);
    }
    for (; column + 16 <= column_end; column += 16) {
        quantize_bf16_s8_per_channel_16(
                src, src_lda, dst, dst_lda, scales, rows, column);
    }
    if (column < column_end) {
        quantize_bf16_s8_per_channel_scalar(
                src, src_lda, dst, dst_lda, scales, rows, column, column_end);
    }
}

// Column blocks from every active matrix are flattened into one task list, so a
// single OpenMP team balances columns across all matrices instead of running one
// team per matrix.  Blocks are 64 wide, used when every active K is a multiple
// of 64 and there are enough of them to fill the team; otherwise the rows are
// split instead (dynamic_per_channel_group_quant_rows).
void dynamic_per_channel_group_quant_bf16_s8_native(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int num_threads) {
    const size_t num_ops = M.size();
    const int thread_count = requested_team_size(num_threads);

    bool all_active_ops_divisible_by_64 = true;
    int64_t blocks_of_64 = 0;
    for (size_t op = 0; op < num_ops; ++op) {
        if (M[op] <= 0) continue;
        all_active_ops_divisible_by_64
                = all_active_ops_divisible_by_64 && (K[op] % 64 == 0);
        blocks_of_64 += K[op] / 64;
    }
    if (!all_active_ops_divisible_by_64 || blocks_of_64 < thread_count) {
        dynamic_per_channel_group_quant_rows(
                src, M, K, lda, dst, dst_lda, scales, thread_count);
        return;
    }
    constexpr int64_t column_block = 64;

    std::vector<int64_t> block_prefix(num_ops);
    int64_t total_blocks = 0;
    for (size_t op = 0; op < num_ops; ++op) {
        if (M[op] > 0) {
            total_blocks += (K[op] + column_block - 1) / column_block;
        }
        block_prefix[op] = total_blocks;
    }
    if (total_blocks == 0) return;

    const int team_size
            = static_cast<int>(std::min<int64_t>(thread_count, total_blocks));
#pragma omp parallel for schedule(static) num_threads(team_size)
    for (int64_t task = 0; task < total_blocks; ++task) {
        // block_prefix holds running block counts, so the first entry above
        // task identifies the matrix that owns this block.
        const auto it = std::upper_bound(
                block_prefix.begin(), block_prefix.end(), task);
        const size_t op = static_cast<size_t>(it - block_prefix.begin());
        const int64_t op_block_begin = op == 0 ? 0 : block_prefix[op - 1];
        const int64_t column = (task - op_block_begin) * column_block;
        const int64_t columns_in_block
                = std::min<int64_t>(column_block, K[op] - column);
        const auto *src_data = static_cast<const uint16_t *>(src[op]);
        auto *dst_data = static_cast<int8_t *>(dst[op]);

        dynamic_per_channel_quant_bf16_s8_native(src_data, lda[op], dst_data,
                dst_lda[op], scales[op], M[op], column,
                column + columns_in_block);
    }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
