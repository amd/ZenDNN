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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT8_UKERNEL_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT8_UKERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::data_type_t;

/// Largest K for which the int32 accumulation of a u8 x s8 dot product
/// cannot overflow for any byte values; the zero-point term stays exact up
/// to it for a zero point within the source type's range, and in int64
/// (wide_zp) for any other.
inline constexpr int kMaxSafeInt8AccumK
        = std::numeric_limits<int32_t>::max() / (255 * 128);

/// One MR x NR tile (NR = 16 * nv) over a native-layout INT8 block.
struct native_int8_ukernel_args_t {
    /// MR rows, k_quads * 4 readable bytes each.
    const uint8_t *A = nullptr;
    int lda = 0;
    /// Quad 0 of the tile's first pack column.
    const int8_t *B = nullptr;
    /// Bytes between consecutive K quads of B.
    int b_stride = 0;
    int k_quads = 0;
    const int32_t *col_sum = nullptr; ///< NR
    /// MR zero points, + 128 for a signed A (without it when wide_zp).
    const int32_t *zp = nullptr;
    /// The zero-point term in int64 (the shift for a signed A added there):
    /// exact for any zero point, where the int32 form wraps once K times the
    /// largest |a - zp| times 128 exceeds the int32 range.
    bool wide_zp = false;
    const float *src_scale = nullptr; ///< MR
    const float *wei_scale = nullptr; ///< NR
    const float *bias = nullptr; ///< NR, or nullptr
    float *C = nullptr;
    int ldc = 0;
    /// For a K split into passes: int32 sums of the tile at C's positions
    /// (m * ldc + n).  A pass after the first adds the sums found there; a
    /// pass before the last stores its sums there and leaves C to the last.
    /// nullptr for a call over all of K.
    int32_t *partial = nullptr;
    bool first_pass = true;
    bool last_pass = true;
    /// When set, the result goes here instead of C: MR rows dst_ldc
    /// elements apart, bf16 rounded to nearest even or f32.
    void *dst = nullptr;
    int dst_ldc = 0;
    bool dst_bf16 = false;
};

/// C[m][n] = f32(acc - zp[m] * col_sum[n]) * (src_scale[m] * wei_scale[n]),
/// or one FMA with bias[n], where acc is the int32 dot product of row m and
/// column n (over all passes).  A signed A is read as A + 128.  The int32
/// arithmetic wraps (int64 for the zero-point term with wide_zp) and the f32
/// operations run in exactly this order, so equal inputs give equal bits.
using native_int8_ukernel_fn_t = void (*)(const native_int8_ukernel_args_t &);

/// Largest MR served for NR = 16 * nv (nv 1 or 2); 0 otherwise.
int native_int8_ukernel_max_mr(int nv);

/// nullptr outside MR 1 .. native_int8_ukernel_max_mr(nv).
native_int8_ukernel_fn_t select_native_int8_ukernel(
        int mr, int nv, bool src_signed);

/// A batch-reduce INT8 kernel over one MR x NR tile (NR = 16 * nv): a batch
/// of (A, B) blocks, each reduced exactly in int32, then taken through a
/// per-element step into f32 accumulators that stay in registers across the
/// batch; C enters before the first element and leaves after the last.  The
/// per-group tile runs it with one element per K group.
struct native_int8_group_ukernel_args_t {
    // The batch.  Element g multiplies MR rows of s8 source, at A + g *
    // a_step and lda apart, by B + g * b_step, k_quads K quads each.
    const uint8_t *A = nullptr;
    int lda = 0;
    size_t a_step = 0;
    /// Quad 0 of the tile's first pack column; b_stride bytes between quads.
    /// The weight is read as w + 128.
    const int8_t *B = nullptr;
    int b_stride = 0;
    size_t b_step = 0;
    int k_quads = 0;
    int batch = 0;
    // The step of element g, for row m: the row's int32 sum starts at
    // shift[m * shift_ld + g * shift_step], which cancels what reading the
    // weight as w + 128 adds (-128 times the sum of the row's source values
    // over the element), and the scales are src_scale[m * src_scale_ld + g *
    // src_scale_step] and the NR weight scales from element g *
    // wei_scale_step of wei_scale on, of type wei_scale_dt (f32, bf16 or
    // f16), each read exactly as f32.
    const int32_t *shift = nullptr;
    int shift_ld = 0;
    int shift_step = 0;
    const float *src_scale = nullptr;
    int src_scale_ld = 0;
    int src_scale_step = 0;
    const void *wei_scale = nullptr;
    int wei_scale_step = 0;
    data_type_t wei_scale_dt = data_type_t::f32;
    /// The f32 tile, ldc apart.  With accumulate it holds the sum of the
    /// elements before this batch; else the sum starts at +0.0.
    float *C = nullptr;
    int ldc = 0;
    bool accumulate = false;
    /// When set, the sum is stored here instead of C: MR rows dst_ldc
    /// elements apart, bf16 rounded to nearest even or f32.
    void *dst = nullptr;
    int dst_ldc = 0;
    bool dst_bf16 = false;
};

/// C[m][n] (+)= the sum over the batch, in order, of (f32(shift_g + acc_g)
/// * src_scale_g) * wei_scale_g[n], where acc_g is the int32 dot product of
/// row m of element g's A and column n of its B read as w + 128: one
/// rounding per operation and no fused multiply-add, the first element
/// added to the entry value of C (+0.0 without accumulate).  shift_g + acc_g
/// is the exact sum of element g, so equal inputs give equal bits.
using native_int8_group_ukernel_fn_t
        = void (*)(const native_int8_group_ukernel_args_t &);

/// Largest MR served for NR = 16 * nv: 12 at nv 1, 6 at nv 2, 0 otherwise.
/// The int32 and f32 accumulators, the B vectors, the broadcast source, the
/// flip constant and the step's scales fit the 32 vector registers.
int native_int8_group_ukernel_max_mr(int nv);

/// nullptr outside MR 1 .. native_int8_group_ukernel_max_mr(nv).
native_int8_group_ukernel_fn_t select_native_int8_group_ukernel(int mr, int nv);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_INT8_NATIVE_INT8_UKERNEL_HPP
