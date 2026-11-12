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
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// K values of an s4 weight in one K group of the layout: an octet, whose
/// 4 bytes per column hold two K quads.
inline constexpr int kNativeS4Octet = 8;

/// K groups of a weight in the layout: K pairs for bf16, K quads for s8, K
/// octets for s4, K rows for f32 and f16.  No intermediate overflows, at any
/// K.
inline int native_k_groups(data_type_t wei_dt, int K) {
    if (wei_dt == data_type_t::f32 || wei_dt == data_type_t::f16) return K;
    const int g = wei_dt == data_type_t::s8 ? INT8_VNNI_GRP
            : wei_dt == data_type_t::s4     ? kNativeS4Octet
                                            : VNNI_PAIR;
    return K / g + (K % g != 0);
}

/// Bytes of one block of pack_nr columns: the K groups, then for s8 the
/// pack_nr int32 column sums.
inline size_t native_block_bytes(data_type_t wei_dt, int K, int pack_nr) {
    const size_t groups = static_cast<size_t>(native_k_groups(wei_dt, K));
    if (wei_dt == data_type_t::s8)
        return (groups * INT8_VNNI_GRP + sizeof(int32_t)) * pack_nr;
    if (wei_dt == data_type_t::f32) return groups * sizeof(float) * pack_nr;
    if (wei_dt == data_type_t::f16) return groups * sizeof(uint16_t) * pack_nr;
    if (wei_dt == data_type_t::s4) return groups * INT8_VNNI_GRP * pack_nr;
    return groups * VNNI_PAIR * sizeof(uint16_t) * pack_nr;
}

/// The columns the layout holds for a weight of N columns: N rounded up to
/// whole blocks of pack_nr (N itself when pack_nr is not positive).  The
/// columns from N on hold zeros.
inline int native_padded_n(int N, int pack_nr) {
    if (pack_nr <= 0) return N;
    return (N + pack_nr - 1) / pack_nr * pack_nr;
}

/// Whether the layout holds a K x N weight of wei_dt in blocks of pack_nr
/// columns: bf16, s8, s4, f32 or f16, K >= 1, N from 1 to kMaxNativeN and
/// pack_nr == kNativeLayoutPackNr.  A weight of N not a multiple of pack_nr
/// is held in native_padded_n(N, pack_nr) columns, zero from N on.  The
/// layout rule (decide_native_layout) admits no other weight; the functions
/// of the layout, the weight view and the looper assume it.
inline bool native_layout_fits(data_type_t wei_dt, int K, int N, int pack_nr) {
    return (wei_dt == data_type_t::bf16 || wei_dt == data_type_t::s8
                   || wei_dt == data_type_t::s4 || wei_dt == data_type_t::f32
                   || wei_dt == data_type_t::f16)
            && K >= 1 && pack_nr == kNativeLayoutPackNr && N >= 1
            && N <= kMaxNativeN;
}

/// Bytes of a K x N weight in the layout (see native_weight_view_t), its
/// padded columns included, rounded up to 64, for a weight
/// native_layout_fits takes.
size_t native_packed_bytes(data_type_t wei_dt, int K, int N, int pack_nr);

/// Writes weight -- [K, N] row-major ldb apart, or [N, K] when transB --
/// into dst (native_packed_bytes) in the layout, splitting the blocks over up
/// to num_threads threads (one inside a parallel region).  Odd K pads a
/// pair, K % 4 != 0 a quad, K % 8 != 0 an s4 octet, with zeros (f32 and f16
/// weights have no padding);
/// s8 column sums cover the real K.  An s4 weight holds two's-complement
/// nibbles at its linear element index (n * ldb + k with transB, else
/// k * ldb + n, ldb in elements), the even index in the low nibble of its
/// byte.  The pack columns from N to native_padded_n(N, pack_nr) are zero,
/// s8 column sums included, and no byte of the weight past its N columns is
/// read.  For a weight native_layout_fits takes, on a CPU with AVX-512 F, BW
/// and VL (native_base_isa); failure for a null weight or dst, or an ldb
/// below the row length.  The layout depends on the weight alone: the
/// gated tiles of the native BRGEMM pair the halves of a gated call
/// (native_weight_view_t).
status_t native_pack_weight(data_type_t wei_dt, const void *weight, int K,
        int N, int ldb, int pack_nr, bool transB, void *dst, int num_threads);

/// The tiles native_pack_weight splits a bf16, f32 or f16 weight stored
/// [K, N] into, for n_blocks whole blocks over k_groups K groups (K pairs for
/// bf16, K rows for f32 and f16): 0 when their count passes the int range
/// the threads' loop counts in, and such a weight packs block by block.
int native_pack_tiles(int n_blocks, int k_groups);

/// Inverse of native_pack_weight for bf16: writes every weight element the
/// image packed back to weight (same arguments as the pack).  The padding of
/// a padded ldb is not written.  For an image of N a multiple of pack_nr,
/// the only images an in-place rewrite makes.
void native_unpack_bf16(const void *packed, int K, int N, int ldb, int pack_nr,
        bool transB, void *weight, int num_threads);

/// native_unpack_bf16 for an f16 image (same arguments and contract).
void native_unpack_f16(const void *packed, int K, int N, int ldb, int pack_nr,
        bool transB, void *weight, int num_threads);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_PACK_HPP
