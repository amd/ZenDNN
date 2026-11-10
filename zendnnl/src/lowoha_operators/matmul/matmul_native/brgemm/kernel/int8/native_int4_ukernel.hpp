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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT4_UKERNEL_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT4_UKERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::data_type_t;

/// Largest group of K values over which the int32 sum of an s8 source
/// times an s4 weight is exact: |sum| <= 128 * 8 * group.
inline constexpr int kMaxSafeInt4GroupK
        = std::numeric_limits<int32_t>::max() / (128 * 8);

/// Largest K of an s4 weight whose K padded to whole octets, which the
/// layout's K groups and a call's source rows count, an int holds.
inline constexpr int kMaxNativeInt4K
        = std::numeric_limits<int32_t>::max() / 8 * 8;

/// A batch-reduce kernel over one MR x NR tile (NR = 16 * nv) of an s8
/// source and an s4 weight in the native layout (native_weight_view_t): a
/// batch of K groups, each reduced exactly in int32, then taken into f32
/// accumulators that stay in registers across the batch; C enters before
/// the first group and leaves after the last.
struct native_int4_group_ukernel_args_t {
    // Group g multiplies MR rows of s8 source, at A + g * a_step and lda
    // apart, k_quads K quads each, by the weight's K quads from quad0 + g *
    // k_quads on: quad q is the low nibbles (q even) or the high nibbles of
    // the octet at B + (q / 2) * b_stride, read as w + 8.
    const uint8_t *A = nullptr;
    int lda = 0;
    size_t a_step = 0;
    const uint8_t *B = nullptr;
    int b_stride = 0;
    /// The tile's second 16 columns (nv 2): their octet b_second bytes after
    /// B's, their weight scales scale_second elements after the first 16
    /// columns'; 64 and 16 inside one block, or where a paired tile's up run
    /// starts.
    size_t b_second = 64;
    int scale_second = 16;
    int quad0 = 0; ///< 0 or 1
    int k_quads = 0;
    int batch = 0;
    // The sum of row m and group g starts at shift[m * shift_ld + g *
    // shift_step], -8 times the sum of the row's source values over the
    // group, which cancels what reading the weight as w + 8 adds.  The
    // scales are src_scale[m * src_scale_ld + g * src_scale_step] and the
    // NR weight scales from element g * wei_scale_step of wei_scale on, of
    // type wei_scale_dt (f32, bf16 or f16), each read exactly as f32.  The
    // row strides are size_t: m times the groups of a large K exceeds an
    // int.
    const int32_t *shift = nullptr;
    size_t shift_ld = 0;
    int shift_step = 0;
    const float *src_scale = nullptr;
    size_t src_scale_ld = 0;
    int src_scale_step = 0;
    const void *wei_scale = nullptr;
    int wei_scale_step = 0;
    data_type_t wei_scale_dt = data_type_t::f32;
    /// The f32 tile, ldc apart.  With accumulate it holds the sum of the
    /// groups before this batch; else the sum starts at +0.0.
    float *C = nullptr;
    int ldc = 0;
    bool accumulate = false;
    /// The fused order's last batch of a source scale per row: the sum is
    /// multiplied by the row's source scale after it.
    bool finish = false;
    /// K passes of a batch of one group: with partial_in its int32 sum
    /// starts at the int32 values C holds (the earlier passes' sum, shift
    /// included) instead of the shift; with partial_out the sum is stored
    /// to C as int32, and neither a scale nor dst is applied.  accumulate
    /// is then false.
    bool partial_in = false;
    bool partial_out = false;
    /// When set, the result is stored here instead of C: MR rows dst_ldc
    /// elements apart, bf16 rounded to nearest even or f32.
    void *dst = nullptr;
    int dst_ldc = 0;
    bool dst_bf16 = false;
};

/// C[m][n] (+)= the batch, in order, of each group's exact int32 sum S_g of
/// row m times column n (shift_g plus the dot product of the source and the
/// weight read as w + 8), from the entry value of C (+0.0 without
/// accumulate):
///   reference: C + (f32(S_g) * src_scale_g) * wei_scale_g[n], one rounding
///     per operation and no fused multiply-add;
///   fused: fma(f32(S_g), wei_scale_g[n], C), one rounding per group, and
///     with finish then C * src_scale[m * src_scale_ld];
///   fused over source scales per group: fma(f32(S_g), src_scale_g *
///     wei_scale_g[n], C), the product rounded and +0.0 where S_g is 0.
/// Equal inputs give equal bits; a group taken in K passes (partial_in,
/// partial_out) gives the bits of the group taken whole.
using native_int4_group_ukernel_fn_t
        = void (*)(const native_int4_group_ukernel_args_t &);

/// Largest MR served for NR = 16 * nv: 12 at nv 1, 6 at nv 2, 0 otherwise.
/// The int32 and f32 accumulators, the two nibble planes of a load, the two
/// broadcast source quads and the nibble mask fit the 32 vector registers.
int native_int4_group_ukernel_max_mr(int nv);

/// nullptr outside MR 1 .. native_int4_group_ukernel_max_mr(nv); src_groups:
/// the fused order over source scales per group.
native_int4_group_ukernel_fn_t select_native_int4_group_ukernel(
        int mr, int nv, native_int8_order_t order, bool src_groups = false);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT4_UKERNEL_HPP
