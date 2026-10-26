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

#include <cstdint>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// What an INT8 call does where its scheme and order give a choice.
struct native_int8_decision_t {
    native_int8_scheme_t scheme = native_int8_scheme_t::per_channel;
    int groups = 0; ///< G of a per-group call, 0 otherwise
    native_int8_order_t order = native_int8_order_t::fused;
    /// The bias may move into the dequantization (native_int8_call_t::bias).
    bool bias_in_dequant = false;
    /// The kernel may store the destination itself when the epilogue has
    /// nothing else to do.
    bool direct_store = false;
};

/// The decision of an INT8 call with weight scales params.quant_params and
/// bias (params.dtypes.bias, or nullptr):
///   * per_group when int8_scale_groups(params, desc.N) > 0, else
///     per_channel;
///   * per_channel: the reference order with alpha other than 1, beta other
///     than 0 or a u8, s8 or s32 destination, which rounds the value to an
///     integer; the fused order otherwise;
///   * per_group: the reference order, alpha and beta after the groups (in
///     the epilogue);
///   * bias_in_dequant: a per-channel call of the fused order with an f32,
///     bf16 or f16 bias;
///   * direct_store: a per-channel call of the fused order with an f32 or
///     bf16 destination.
native_int8_decision_t decide_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias);

/// Per-call INT8 operands, prepared once before the tile loop.
struct native_int8_call_t {
    /// Source rows of K bytes zero padded to whole quads, lda apart: u8, or
    /// s8 when src_signed, which the tile reads as s8 + 128 and compensates
    /// with a zero point raised by 128.
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
    /// adds (f32(sum) * src scale) * wei scale over the groups in order;
    /// alpha, beta and the bias are the epilogue's.
    native_int8_scheme_t scheme = native_int8_scheme_t::per_channel;
    int groups = 0;
    int group_quads = 0;
    int src_scale_group_stride = 0;
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
    /// A dynamically quantized source and the scales and zero points the
    /// quantizer allocated.
    reorder_quant_buffers_t src_buffers;
};

/// The INT8 tile serves the call's weight, source and quantization params
/// with decision, decide_int8_call's for them (unimplemented otherwise).
/// Does no work and reads no buffer, so a call it rejects can still take
/// another route.  Weight zero points, K above kMaxSafeInt8AccumK and a
/// gated activation with beta != 0 are not served.  Per-group weight scales
/// need groups of whole K quads, an s8 source without a zero point (or a
/// bf16 / f32 source quantized to s8), a per-tensor, per-token or {M, G}
/// source scale and an f32 / bf16 dst.
status_t check_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision);

/// G of a per-group {G > 1, N} weight scale, 0 for any other.
int int8_scale_groups(const matmul_params &params, int N);

/// Quantizes a bf16 / f32 source per token or per tensor (or per group, with
/// per-group weight scales) with matmul's reorder_quantization_wrapper
/// (scales and zero points written back to the caller's buffers when it gave
/// them), or adopts a u8 / s8 source with its per-tensor, per-token or
/// per-group scale and zero point, for a call check_int8_call accepts with
/// decision.  The source is copied only when it is transposed or K is not
/// whole quads.  The copies (and the transpose before a dynamic
/// quantization) work in blocks of copy_block rows (by copy_block columns
/// when they transpose), on the requested threads when M * K >=
/// copy_par_min.
status_t prepare_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        const void *src, int copy_block, int64_t copy_par_min,
        native_int8_call_t &call);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_INT8_CALL_HPP
