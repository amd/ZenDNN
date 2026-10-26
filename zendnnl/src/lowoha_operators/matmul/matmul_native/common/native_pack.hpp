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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_PACK_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_PACK_HPP

#include <cstddef>
#include <cstdint>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// Pack widths (columns per block) a native-layout weight can have.
inline constexpr int kNativeMinPackNr = 32;
inline constexpr int kNativeMaxPackNr = 64;

/// K groups of a weight in the layout: K pairs for bf16, K quads for s8.
inline int native_k_groups(data_type_t wei_dt, int K) {
    const int g = wei_dt == data_type_t::s8 ? INT8_VNNI_GRP : VNNI_PAIR;
    return (K + g - 1) / g;
}

/// Bytes of one block of pack_nr columns: the VNNI groups, then for s8 the
/// pack_nr int32 column sums.
inline size_t native_block_bytes(data_type_t wei_dt, int K, int pack_nr) {
    const size_t groups = static_cast<size_t>(native_k_groups(wei_dt, K));
    if (wei_dt == data_type_t::s8)
        return (groups * INT8_VNNI_GRP + sizeof(int32_t)) * pack_nr;
    return groups * VNNI_PAIR * sizeof(uint16_t) * pack_nr;
}

/// Bytes of a K x N weight in the layout (see native_weight_view_t) rounded
/// up to 64; 0 for a weight type other than bf16 / s8, K < 1, a pack width
/// other than kNativeMinPackNr / kNativeMaxPackNr, or N not a positive
/// multiple of it.
size_t native_packed_bytes(data_type_t wei_dt, int K, int N, int pack_nr);

/// Writes weight -- [K, N] row-major ldb apart, or [N, K] when transB --
/// into dst (native_packed_bytes) in the layout, splitting the blocks over up
/// to num_threads threads (one inside a parallel region).  With
/// interleave_split_halves, pack column 2i holds column i and pack column
/// 2i + 1 holds column N / 2 + i.  Odd K pads a pair, K % 4 != 0 a quad,
/// with zeros; s8 column sums cover the real K.
status_t native_pack_weight(data_type_t wei_dt, const void *weight, int K,
        int N, int ldb, int pack_nr, bool transB, bool interleave_split_halves,
        void *dst, int num_threads);

/// Inverse of native_pack_weight for bf16: writes every weight element the
/// image packed back to weight (same arguments as the pack).  The padding of
/// a padded ldb is not written.
void native_unpack_bf16(const void *packed, int K, int N, int ldb, int pack_nr,
        bool transB, bool interleave_split_halves, void *weight,
        int num_threads);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_PACK_HPP
