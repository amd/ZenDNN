/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP

#include <cstddef>
#include <cstdint>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// A weight in the native VNNI layout, read in place.  Per block of
/// pack_nr columns:
///   bf16: [ceil(K / 2)][pack_nr][2] bf16, odd K padded with zero.
///   s8:   [ceil(K / 4)][pack_nr][4] s8, zero padded, then [pack_nr] int32
///         column sums over the real K.
/// A gate / up weight packed with interleave_split_halves holds gate column i
/// at pack column 2i and up column i at pack column 2i + 1.
struct native_weight_view_t {
    const uint8_t *base = nullptr;
    data_type_t wei_dt = data_type_t::none;
    int K = 0;
    int N = 0;
    int pack_nr = 0;
    int k_groups = 0; ///< K pairs (bf16) or K quads (s8)
    size_t block_bytes = 0; ///< one block, column sums included
    bool interleaved = false;

    const uint8_t *block(int nb) const {
        return base + static_cast<size_t>(nb) * block_bytes;
    }

    /// s8 only: the column sums of block nb.
    const int32_t *col_sums(int nb) const {
        return reinterpret_cast<const int32_t *>(
                block(nb) + static_cast<size_t>(k_groups) * pack_nr * 4);
    }

    /// Elements of wei_dt between consecutive K groups of one column.
    int group_stride() const {
        return pack_nr * (wei_dt == data_type_t::bf16 ? 2 : 4);
    }

    /// Weight column held at pack column p.
    int logical_col(int p) const {
        return interleaved ? ((p & 1) ? N / 2 + p / 2 : p / 2) : p;
    }
};

/// How the call's weight reaches the native BRGEMM.
struct native_weight_ref_t {
    /// The raw weight, an AOCL prepack (aocl_blocked), or an image an earlier
    /// native call rewrote in place.
    const void *weight = nullptr;
    int pack_nr = 0;
    /// silu_and_mul / gelu_and_mul: gate | up halves interleaved in the pack.
    bool interleave_split_halves = false;
    /// Effective weight cache mode (see get_native_layout): 0 packs for this
    /// call only, 1 uses the native layout cache, 2 rewrites the buffer in
    /// place when the image fits it (a bf16 raw weight, or an AOCL prepack),
    /// else as 1.
    int32_t weight_cache_mode = 1;
    /// Writable bytes at weight (0: the logical extent, or for an AOCL
    /// prepack its size).
    size_t capacity_bytes = 0;
    /// The buffer holds the weight as an aocl_dlp_blocked prepack (or an
    /// image an earlier native call rewrote in place): the raw weight is its
    /// AOCL inverse, rows transB ? K : N elements apart.
    bool aocl_blocked = false;
};

/// Owns a pack made for one call and frees it on scope exit.
class native_weight_lease_t {
public:
    native_weight_lease_t() = default;
    ~native_weight_lease_t();
    native_weight_lease_t(const native_weight_lease_t &) = delete;
    native_weight_lease_t &operator=(const native_weight_lease_t &) = delete;

    void own(void *packed);

private:
    void *packed_ = nullptr;
};

/// Resolves ref to a view through get_native_layout under its weight cache
/// mode; a missing pack uses up to num_threads threads.  Fails rather than
/// guessing when the buffer cannot be read as the layout.
status_t acquire_native_weight(const native_weight_ref_t &ref,
        data_type_t wei_dt, int K, int N, int ldb, bool transB,
        native_weight_view_t &view, native_weight_lease_t &lease,
        int num_threads = 1);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_WEIGHT_VIEW_HPP
