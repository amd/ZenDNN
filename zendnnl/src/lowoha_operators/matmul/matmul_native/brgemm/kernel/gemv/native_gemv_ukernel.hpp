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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_GEMV_NATIVE_GEMV_UKERNEL_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_GEMV_NATIVE_GEMV_UKERNEL_HPP

#include <cstddef>
#include <cstdint>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::data_type_t;

/// Blocks of 32 columns one GEMV kernel call reads side by side at most:
/// each block a stream of its own, all of them fed by one load of the source
/// per K group.
inline constexpr int kNativeGemvMaxBlocks = 4;

/// The blocks of an INT8 or s4 kernel call at most: their sums take more
/// accumulators per block (two sets, four for one block).
inline constexpr int kNativeGemvMaxIntBlocks = 2;

/// The columns of one GEMV kernel call over a weight in the native layout
/// (native_weight_view_t): nb blocks of 32 columns, whose K groups are read in
/// K order.  Block i's first 16 columns have their K group 0 at lo[i]; its
/// second 16 follow them in the block (64 bytes on, 32 for an f16 weight), or
/// with runs are the 16 columns at hi[i] (the up run of a gated call's [gate |
/// up] halves, which may lie in another block).
struct native_gemv_cols_t {
    int nb = 0;
    bool runs = false;
    const uint8_t *lo[kNativeGemvMaxBlocks] = {};
    const uint8_t *hi[kNativeGemvMaxBlocks] = {};
};

/// The GEMV kernels compute the 32 * nb columns of the call for one source
/// row over all of K into c (or s), block i's columns at 32 * i: each column
/// by the steps of the tile kernels of its type in K order, so equal inputs
/// give the tiles' bits for every M = 1 call.  Bytes between K groups: 128
/// (bf16 pairs, s8 quads, s4 octets, f32 rows), 64 (f16 rows).

/// BF16: from +0.0, one dot product of a K pair per pair in K order, an odd
/// K's last element paired with zero (bf16_brgemm_fn_t's accumulation).  a
/// holds K bf16.
using native_bf16_gemv_fn_t = void (*)(const uint16_t *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c);

/// FP32: from +0.0, one fused multiply-add per k in K order
/// (native_fp32_ukernel_fn_t).  a holds K f32.
using native_f32_gemv_fn_t = void (*)(const float *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c);

/// F16 over a source widened to f32 (a, K f32): each weight element widened
/// exactly, then as FP32 (native_f16_ukernel_fn_t).
using native_f16_gemv_fn_t = void (*)(const float *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c);

/// F16_PH over the f16 source (a, K f16): blocks of kNativeF16AccumBlockK
/// steps from k = 0, each from +0.0 with one f16 fused multiply-add per k,
/// widened to f32 exactly and added to the f32 sum, which starts at +0.0
/// (native_f16_ph_ukernel_fn_t).
using native_f16_ph_gemv_fn_t = void (*)(const uint16_t *__restrict a, int K,
        const native_gemv_cols_t &cols, float *__restrict c);

/// INT8 per-channel: s[32 * i + j] = the sum over k_quads K quads of the dot
/// products of the source's quad, u8, or s8 read as a + 128 (src_signed), and
/// the s8 weight quad of the column, wrapping in int32: the sum the
/// per-channel INT8 kernel (native_int8_ukernel_fn_t) finishes.  a holds
/// k_quads * 4 bytes.
using native_s8_gemv_fn_t = void (*)(const uint8_t *__restrict a, int k_quads,
        const native_gemv_cols_t &cols, int32_t *__restrict s);

/// INT8 per-channel in the fused order: the sums of native_s8_gemv_fn_t
/// finished as the per-channel INT8 kernel finishes them, c[32 * i + j] =
/// f32(S - zp' * cs) * (src_scale * wei_scale) or one fused multiply-add
/// with the bias, where zp' is the zero point plus 128 for an s8 source and
/// the zero-point term is taken in int64 with wide_zp: the f32 values the
/// tiles give before their epilogue.
struct native_s8_gemv_args_t {
    /// The source row, k_quads * 4 bytes: u8, or s8 (src_signed).
    const uint8_t *a = nullptr;
    int k_quads = 0;
    native_gemv_cols_t cols;
    /// The column sums of block i's first and second 16 columns, and their
    /// weight columns, where the weight scales and the bias are read.
    const int32_t *cs_lo[kNativeGemvMaxBlocks] = {};
    const int32_t *cs_hi[kNativeGemvMaxBlocks] = {};
    int col_lo[kNativeGemvMaxBlocks] = {};
    int col_hi[kNativeGemvMaxBlocks] = {};
    /// The source's zero point, without the 128 of an s8 source.
    int32_t zp = 0;
    bool wide_zp = false;
    float src_scale = 0.0f;
    /// f32, bf16 or f16 weight scales, per column or one per tensor, and an
    /// f32, bf16 or f16 bias per column (or none), read exactly as f32.
    const void *wei_scale = nullptr;
    data_type_t wei_scale_dt = data_type_t::f32;
    bool wei_scale_per_column = true;
    const void *bias = nullptr;
    data_type_t bias_dt = data_type_t::f32;
    float *c = nullptr;
};

using native_s8_fused_gemv_fn_t = void (*)(const native_s8_gemv_args_t &);

/// A GEMV kernel call over K groups of an s8 weight with per-group weight
/// scales, or of an s4 weight: per group the exact int32 sum S_g, then the
/// group kernel's step of the call's order into the f32 sum of each column
/// (native_int8_group_ukernel_fn_t, native_int4_group_ukernel_fn_t).
struct native_gemv_group_args_t {
    /// The source row: group g's K values at a + g * a_step.  s8, which an s8
    /// weight reads as w + 128 and an s4 weight as w + 8, its sums starting at
    /// shift[g] to cancel that; or (an s8 weight only) u8, the sum less src_zp
    /// times the group's column sums, that term in int64 with wide_zp.
    const uint8_t *a = nullptr;
    size_t a_step = 0;
    bool src_u8 = false;
    const int32_t *shift = nullptr;
    int32_t src_zp = 0;
    bool wide_zp = false;
    native_gemv_cols_t cols;
    /// Bytes from a group's first K group of a column to the next group's.
    size_t b_step = 0;
    int k_quads = 0; ///< K quads per group
    /// An s4 weight: the first group's first quad, counted from the octets at
    /// cols: 0 or 1 (the high nibbles).
    int quad0 = 0;
    int groups = 0;
    /// Group g's source scale at src_scale[g * src_scale_step].
    const float *src_scale = nullptr;
    int src_scale_step = 0;
    /// Group g's weight scale of weight column n at element g * wei_scale_ld
    /// + n of wei_scale, of type wei_scale_dt (f32, bf16 or f16), read
    /// exactly as f32; col_lo[i] and col_hi[i] are the weight columns of block
    /// i's first and second 16 columns.  unit_scales: every weight scale is
    /// 1, and over an s4 weight every source scale too.
    const void *wei_scale = nullptr;
    data_type_t wei_scale_dt = data_type_t::f32;
    size_t wei_scale_ld = 0;
    int col_lo[kNativeGemvMaxBlocks] = {};
    int col_hi[kNativeGemvMaxBlocks] = {};
    bool unit_scales = false;
    /// An s4 weight's order: fused, fma(f32(S_g), wei scale, C), then with
    /// finish C times the source scale; fused over source scales per group
    /// (src_groups), fma(f32(S_g), source scale * wei scale, C), the product
    /// zero where S_g is; reference, C + (f32(S_g) * source scale) * wei
    /// scale.  An s8 weight takes the reference order.
    native_int8_order_t order = native_int8_order_t::reference;
    bool src_groups = false;
    bool finish = false;
    /// The sums of the 32 * nb columns, from +0.0, block i's at 32 * i.
    float *c = nullptr;
};

using native_gemv_group_fn_t = void (*)(const native_gemv_group_args_t &);

/// The kernels of nb blocks, 1 .. kNativeGemvMaxBlocks (INT8 and s4: 1 ..
/// kNativeGemvMaxIntBlocks), with runs or not; nullptr for any other nb.
/// BF16 needs AVX512-BF16, INT8 AVX512-VNNI, F16_PH AVX512-FP16 (null in a
/// build whose compiler lacks its intrinsics), and all AVX-512 F, BW, VL, DQ
/// and FMA.
native_bf16_gemv_fn_t select_native_bf16_gemv(int nb, bool runs);
native_f32_gemv_fn_t select_native_f32_gemv(int nb);
native_f16_gemv_fn_t select_native_f16_gemv(int nb);
native_f16_ph_gemv_fn_t select_native_f16_ph_gemv(int nb);
native_s8_gemv_fn_t select_native_s8_gemv(int nb, bool runs, bool src_signed);
native_s8_fused_gemv_fn_t select_native_s8_fused_gemv(
        int nb, bool runs, bool src_signed);
/// s4: an s4 weight's group kernel (an s8 source), else an s8 weight's over a
/// u8 source (src_u8) or an s8 one.
native_gemv_group_fn_t select_native_group_gemv(
        int nb, bool runs, bool s4, bool src_u8);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_GEMV_NATIVE_GEMV_UKERNEL_HPP
