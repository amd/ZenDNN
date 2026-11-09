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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_CALL_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_CALL_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// The decision of an INT8 call with weight scales params.quant_params and
/// bias (params.dtypes.bias, or nullptr):
///   * an s4 weight: per_group, of the G of its {G, N} weight scales
///     (int4_scale_groups; G 1 is one group over K); the fused order, with a
///     direct store, at alpha 1 and beta 0 into an f32 or bf16 dst; the
///     reference order otherwise, over one group with alpha other than 1 or
///     beta other than 0 with the scales after the group
///     (scales_after_groups);
///   * per_group when int8_scale_groups(params, desc.N) > 0, or, with
///     per-tensor or per-channel weight scales (or none), when
///     int8_source_groups(params, desc.M, desc.N) > 0 (scales_after_groups);
///     else per_channel;
///   * per_channel: the reference order with alpha other than 1, beta other
///     than 0 or a u8, s8 or s32 destination, which rounds the value to an
///     integer; the fused order otherwise;
///   * per_group: the reference order, alpha and beta after the groups (in
///     the epilogue, or with scales_after_groups in the tile, before the
///     weight scales);
///   * bias_in_dequant: a per-channel call of the fused order with an f32,
///     bf16 or f16 bias;
///   * direct_store: a per-channel call of the fused order, or a per-group
///     call with per-group weight scales, alpha 1 and beta 0, with an f32 or
///     bf16 destination.
native_int8_decision_t decide_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias);

/// Per-call INT8 operands, prepared once before the tile loop.
struct native_int8_call_t {
    /// Source rows of K bytes zero padded to whole quads (whole octets for
    /// an s4 weight), lda apart: u8, or s8 when src_signed, which the
    /// per-channel tile reads as s8 + 128 (compensated by a zero point raised
    /// by 128) and the group kernels read as it is (the weight read as w +
    /// 128, or an s4 weight as w + 8, compensated by group_shift).  The group
    /// kernel reads a u8 source and the weight as they are and takes the
    /// zero point times each group's column sums of the weight.
    const uint8_t *A = nullptr;
    int lda = 0;
    bool src_signed = false;
    /// Per row (stride 1) or per tensor (stride 0).
    const float *src_scale = nullptr;
    int src_scale_stride = 0;
    /// Per row or per tensor; nullptr when zero.
    const int32_t *src_zp = nullptr;
    int src_zp_stride = 0;
    /// Weight column order; per column (stride 1) or per tensor (stride 0).
    /// f32, bf16 or f16, read in place: a call reads only its tiles' columns.
    const void *wei_scale = nullptr;
    data_type_t wei_scale_dt = data_type_t::f32;
    int wei_scale_stride = 0;
    /// decide_int8_call's.  per_group: the weight scales are per group of
    /// group_quads K quads, element (g, n) at g * N + n, and the source scale
    /// of row m and group g is at src_scale_stride * m +
    /// src_scale_group_stride * g.  The tile sums each group exactly, then
    /// adds (f32(sum) * src scale) * wei scale over the groups in order (in
    /// the fused order of an s4 weight, fma(f32(sum), wei scale, C) over the
    /// groups, then C * src scale, or, with src_scale_group_stride, fma(
    /// f32(sum), src scale * wei scale, C), the product zero where the sum
    /// is zero); alpha, beta and the bias are the epilogue's, unless:
    native_int8_scheme_t scheme = native_int8_scheme_t::per_channel;
    /// decide_int8_call's: the group kernel takes unit weight scales, and
    /// the tile applies alpha, beta and then wei_scale (per column or per
    /// tensor) to the sum of the groups, one rounding each, in the reference
    /// algorithm's order: (alpha * S + beta * C) * wei_scale.  Over an s4
    /// weight of one group the kernel takes unit source scales too, and the
    /// tile applies the source scale before wei_scale: ((alpha * S + beta *
    /// C) * src_scale) * wei_scale.
    bool scales_after_groups = false;
    int groups = 0;
    int group_quads = 0;
    int src_scale_group_stride = 0;
    /// per_group, s8 source: -(1 << shift_bits) times the sum of row m's
    /// source values over group g, at m * groups + g: it cancels what the
    /// group kernel's reading of the weight as w + 128 (an s8 weight,
    /// shift_bits 7) or w + 8 (an s4 weight, shift_bits 3) adds to the
    /// group's sum.  Written by ready_group_shifts.
    int32_t *group_shift = nullptr;
    int shift_bits = 7;
    /// per_group, s8 source: the rows, claimed shift_chunk at a time;
    /// shift_rows[0] is the next chunk a tile may claim, shift_rows[1 + c] is
    /// set once the shifts of chunk c are written.
    int rows = 0;
    int shift_chunk = 1;
    std::unique_ptr<std::atomic<int>[]> shift_rows;
    /// Weight column order (f32, bf16 or f16); set when the dequantization
    /// takes the bias (bias_in_dequant, when the plan allows it).
    const void *bias = nullptr;
    data_type_t bias_dt = data_type_t::f32;
    /// From the descriptor; a per-channel call applies them in its tile.
    float alpha = 1.0f;
    float beta = 0.0f;
    /// decide_int8_call's.  reference, per channel: the kernel yields f32(S),
    /// and alpha, beta, the zero-point term and the scales take one rounding
    /// each in the reference algorithm's order, ((alpha * S + beta * C + Z) *
    /// src_scale) * wei_scale, S the raw integer sum, C the destination read
    /// as float, Z the zero-point term; the bias is the epilogue's.
    native_int8_order_t order = native_int8_order_t::fused;
    /// The C a nonzero beta of a per-channel call reads, element (m, n) at
    /// m * ldc + n; the caller sets it after prepare_int8_call.
    const void *dst = nullptr;
    data_type_t dst_dt = data_type_t::none;
    int ldc = 0;
    /// The zero points can make the int32 zero-point term wrap: the kernels
    /// compute it in int64.
    bool wide_zp = false;
    std::vector<uint8_t> a_buf;
    std::vector<float> scale_buf;
    std::vector<int32_t> zp_buf;
    std::unique_ptr<int32_t[]> shift_buf;
    /// A dynamically quantized source and the scales and zero points the
    /// quantizer allocated.
    native_quant_buffers_t src_buffers;
};

/// The INT8 tile serves the call's weight, source and quantization params
/// with decision, decide_int8_call's for them (unimplemented otherwise, with
/// *why, when why is not null, naming the reason).  Does no work and reads
/// no buffer.  Weight zero points, K above kMaxSafeInt8AccumK, and a gated
/// activation with beta over a dst of fewer than N columns are not served.
/// Per-group weight scales
/// need groups of whole K quads, an f32, bf16, u8, s8 or s32 dst, and an s8
/// source without a zero point (or a bf16 / f32 source quantized to s8) with
/// a per-tensor, per-token or {M, G} source scale, or a u8 source (or a
/// bf16 / f32 source quantized to u8) with a per-tensor scale and zero point.
/// An s4 weight needs K at most kMaxNativeInt4K, {G, N} weight scales (G >=
/// 1) over groups of whole K quads of at most kMaxSafeInt4GroupK values,
/// and an s8 source (given, or a bf16 / f32 source quantized to s8) as
/// per-group weight scales take it: no u8 source, and no source zero point
/// buffer, also with a source the call quantizes.
status_t check_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        const char **why = nullptr);

/// G of a per-group {G > 1, N} weight scale, 0 for any other.
int int8_scale_groups(const matmul_params &params, int N);

/// G of a {G >= 1, N} weight scale, 0 for any other: the groups of an s4
/// weight.
int int4_scale_groups(const matmul_params &params, int N);

/// G of a per-group {M, G > 1} source scale over a weight scale that
/// int8_scale_groups does not take, 0 otherwise; 0 for a null source-scale
/// buffer, unless the call quantizes its source.
int int8_source_groups(const matmul_params &params, int M, int N);

/// Quantizes a bf16 / f32 source per token or per tensor (or per group to
/// s8, with per-group weight scales) with matmul's
/// reorder_quantization_wrapper (scales and zero points written back to the
/// caller's buffers when it gave them), or adopts a u8 / s8 source with its
/// per-tensor, per-token or per-group scale and zero point, for a call
/// check_int8_call accepts with decision.  The source is copied only when it
/// is transposed or K is not whole quads.  The copies (and the transpose
/// before a dynamic quantization) work in blocks of copy_block rows (by
/// copy_block columns when they transpose), on the requested threads when
/// M * K >= copy_par_min.  The group shifts of a per-group call with an s8
/// source are left to its tiles (ready_group_shifts).  A call whose
/// quantization params it cannot use is unimplemented, with *why (when why
/// is not null) naming the reason.
status_t prepare_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        const void *src, int copy_block, int64_t copy_par_min,
        native_int8_call_t &call, const char **why = nullptr);

/// The copy prepare_int8_call makes of a source it does not read in place:
/// dst[m * ldd + k] = byte (m, k) of an M x K source stored row-major lds
/// apart, or column-major when trans; bytes K .. ldd of each row are
/// zeroed.  In blocks of block rows (block columns when trans), on nt
/// threads when M * K >= par_min; any M and K up to the int bound.
void copy_int8_source(const void *src, int lds, bool trans, int M, int K,
        uint8_t *dst, int ldd, int nt, int block, int64_t par_min);

/// Makes the group shifts of rows m0 .. m0 + mr of a prepared per-group call
/// ready: the tiles of a call claim the chunks of rows nobody has claimed
/// yet in order, each chunk once, until every chunk of the range is claimed,
/// then wait for the chunks of the range other tiles are writing.  Safe to
/// call concurrently on any ranges.
void ready_group_shifts(const native_int8_call_t &call, int m0, int mr);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_CALL_HPP
