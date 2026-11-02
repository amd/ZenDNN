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

#ifndef LOWOHA_QUANT_LAYOUT_HPP
#define LOWOHA_QUANT_LAYOUT_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "lowoha_operators/matmul/lowoha_common.hpp"

// Layouts of the scale and zero-point buffers of a matmul: per-tensor,
// per-token (one value per source row), per-channel (one value per output
// column), per-group ({G, N} weight, {M, G} source) and the size of a group
// along K.  matmul_direct's validation, the AOCL-DLP wrapper and the native
// BRGEMM classify quant parameters through these.  A rule that judges a
// buffer by its element count and one that judges it by its dims accept
// different buffers, so each keeps its own helper.

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ── Element counts ──────────────────────────────────────────────────────

/**
 * @brief Checked quant-parameter element count with scalar empty-dims semantics.
 *
 * Quant metadata uses `{}`, `{1}`, and `{1,1}` for per-tensor storage in
 * different frontends. Returns one for empty dims, or -1 for non-positive /
 * overflowing dimensions.
 */
inline int64_t checked_quant_param_num_elements(
        const std::vector<int64_t> &dims) noexcept {
    int64_t count = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim)
            return -1;
        count *= dim;
    }
    return count;
}

// ── Per-tensor ──────────────────────────────────────────────────────────

/** @brief Dims of exactly one element: `{}`, `{1}`, `{1, 1}`, ... */
inline bool quant_param_dims_are_per_tensor(
        const std::vector<int64_t> &dims) noexcept {
    return checked_quant_param_num_elements(dims) == 1;
}

inline bool quant_param_is_per_tensor(
        const matmul_quantization_params_t::matmul_quant_t &q) noexcept {
    return q.buff != nullptr && quant_param_dims_are_per_tensor(q.dims);
}

// ── One value per row or column, by dims ────────────────────────────────

/** @brief A buffer of one value per output column: dims {N} or {1, N}. */
inline bool quant_param_is_per_channel(
        const matmul_quantization_params_t::matmul_quant_t &q, int N) noexcept {
    if (q.buff == nullptr || N <= 0) return false;
    return (q.dims.size() == 1 && q.dims[0] == static_cast<int64_t>(N))
            || (q.dims.size() == 2 && q.dims[0] == 1
                    && q.dims[1] == static_cast<int64_t>(N));
}

/** @brief Dims {M, 1}: one value per source row. */
inline bool quant_param_is_per_token(
        const matmul_quantization_params_t::matmul_quant_t &q, int M) noexcept {
    return M > 0 && q.dims.size() == 2 && q.dims[0] == static_cast<int64_t>(M)
            && q.dims[1] == 1;
}

/**
 * @brief Values of per-tensor or per-row dims: 1 for per-tensor dims, rows
 *        for {rows} or {rows, 1}, 0 for any other dims.
 */
inline int64_t quant_param_tensor_or_row_values(
        const std::vector<int64_t> &dims, int64_t rows) noexcept {
    const int64_t n = checked_quant_param_num_elements(dims);
    if (n == 1) return 1;
    const bool row_shape
            = dims.size() == 1 || (dims.size() == 2 && dims[1] == 1);
    return n == rows && row_shape ? n : 0;
}

// ── Groups along K, by dims ─────────────────────────────────────────────

/** @brief Dims {G > 1, N}: G groups along K of one value per column. */
inline bool quant_param_dims_are_per_group_wei(
        const std::vector<int64_t> &dims, int64_t N) noexcept {
    return dims.size() == 2 && dims[0] > 1 && dims[1] == N;
}

/** @brief True when wei scale dims are {G>1, N} (per-group along K). */
inline bool has_per_group_wei_scale(
        const matmul_quantization_params_t &quant_params, int N) {
    return quant_params.wei_scale.buff != nullptr
            && quant_param_dims_are_per_group_wei(
                    quant_params.wei_scale.dims, static_cast<int64_t>(N));
}

/**
 * @brief True when wei scale dims are {1, N} (per-output-channel).
 *
 * The exact complement of `has_per_group_wei_scale` over 2-D weight
 * scales: that helper requires `dims[0] > 1`, this one `dims[0] == 1`.
 * `sym_quant_group_size` maps it to a single quant group spanning all of K
 * (`K / dims[0]` == `K`), which is what makes it representable by the same
 * sym-quant reorder as the per-group form rather than a distinct layout.
 */
inline bool has_per_channel_wei_scale(
        const matmul_quantization_params_t &quant_params, int N) {
    const auto &wei_dims = quant_params.wei_scale.dims;
    return quant_params.wei_scale.buff != nullptr && wei_dims.size() == 2
            && wei_dims[0] == 1 && wei_dims[1] == static_cast<int64_t>(N);
}

/** @brief Dims {M, groups}, groups > 1: one value per source row and group. */
inline bool quant_param_is_per_group_src(
        const matmul_quantization_params_t::matmul_quant_t &q, int64_t M,
        int64_t groups) noexcept {
    return groups > 1 && q.dims.size() == 2 && q.dims[0] == M
            && q.dims[1] == groups;
}

// ── Layouts by element count ────────────────────────────────────────────

/**
 * @brief Granularity of a quant buffer judged by its element count over the
 *        operand extent it scales: the M rows of a source scale, the N
 *        columns of a weight or destination scale.
 */
enum class quant_count_granularity_t : uint8_t {
    per_tensor = 0, ///< At most one element.
    per_axis = 1, ///< One element per row (source) or column (weight).
    per_group = 2, ///< Any other count: groups along K of those elements.
};

/**
 * @brief The granularity of count elements over extent; one element is
 *        per-tensor whatever the extent.
 */
inline quant_count_granularity_t quant_count_granularity(
        size_t count, int64_t extent) noexcept {
    if (count <= 1) return quant_count_granularity_t::per_tensor;
    return count == static_cast<size_t>(extent)
            ? quant_count_granularity_t::per_axis
            : quant_count_granularity_t::per_group;
}

/**
 * @brief count elements are one per row or column of extent; over an extent
 *        of one, a single element is.
 */
inline bool quant_count_is_per_axis(size_t count, int64_t extent) noexcept {
    return count == static_cast<size_t>(extent);
}

// ── Group size along K ──────────────────────────────────────────────────

/** @brief groups > 0 split K into equal groups. */
inline bool quant_groups_divide(int64_t K, int64_t groups) noexcept {
    return groups > 0 && K % groups == 0;
}

/**
 * @brief groups split K into equal groups of whole quads (4 values): the
 *        groups the sym-quant reorder and the native INT8 group kernel read.
 */
inline bool quant_groups_are_whole_quads(int64_t K, int64_t groups) noexcept {
    return quant_groups_divide(K, groups) && (K / groups) % 4 == 0;
}

/** @brief Groups of group_size values, whole quads, cover K exactly. */
inline bool quant_group_size_is_whole_quads(
        int64_t K, int64_t group_size) noexcept {
    return group_size > 0 && group_size % 4 == 0 && K % group_size == 0;
}

/**
 * @brief K groups of an M-row source scale of count elements, by count: 1
 *        for one value per row, count / M for a whole number of values per
 *        row that splits K into equal groups, 0 for any other count.
 */
inline int64_t quant_count_src_groups(
        int64_t count, int64_t M, int64_t K) noexcept {
    if (M <= 0) return 0;
    if (count == M) return 1;
    if (count <= M || count % M != 0) return 0;
    return quant_groups_divide(K, count / M) ? count / M : 0;
}

/** @brief K-axis B-side sym-quant group size.
 *
 * Prefers wei_scale dims {G, N} → K/G.  Source scale is independent:
 * per-token {M} / {M,1} does not change this value.  Falls back to a
 * src_scale-derived grouping only when wei is not {G, N}.
 */
inline int64_t sym_quant_group_size(
        const matmul_quantization_params_t &quant_params, int M, int N, int K) {
    const auto &wei_dims = quant_params.wei_scale.dims;
    if (wei_dims.size() == 2 && wei_dims[1] == static_cast<int64_t>(N)
            && quant_groups_divide(K, wei_dims[0])) {
        return static_cast<int64_t>(K) / wei_dims[0];
    }

    int64_t src_nelems = quant_params.src_scale.dims.empty() ? 0 : 1;
    for (int64_t d : quant_params.src_scale.dims) {
        src_nelems *= d;
    }
    const int64_t src_groups = (M > 0) ? src_nelems / M : 0;
    return (src_groups > 0) ? static_cast<int64_t>(K) / src_groups
                            : static_cast<int64_t>(K);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_QUANT_LAYOUT_HPP
