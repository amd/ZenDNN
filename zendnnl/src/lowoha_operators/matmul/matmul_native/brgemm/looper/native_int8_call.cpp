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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>

#include <immintrin.h>
#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::common::size_of;

namespace {

using quant_t = matmul_quantization_params_t::matmul_quant_t;

constexpr float kUnitScale = 1.0f;

/// Pauses a wait of ready_group_shifts spins before it yields.
constexpr int kShiftSpinPauses = 4096;

/// The int32 zero-point term of the kernels is exact when every |sum over K
/// of (a - zp) * w| fits int32: K times the largest |a - zp| over the source
/// type's values times 128, for the n zero points of zp.
bool zero_points_fit(const int32_t *zp, int64_t n, bool src_signed, int K) {
    const int64_t lo = src_signed ? -128 : 0;
    const int64_t hi = src_signed ? 127 : 255;
    int64_t dev = std::max(hi, -lo);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t z = zp[i];
        dev = std::max(dev, std::max(z - lo, hi - z));
    }
    return static_cast<int64_t>(K) * dev * 128
            <= std::numeric_limits<int32_t>::max();
}

/// The buffer itself when it is f32, else its n values converted into buf.
const float *f32_values(
        const void *buff, data_type_t dt, int64_t n, std::vector<float> &buf) {
    if (dt == data_type_t::f32) return static_cast<const float *>(buff);
    buf.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        buf[i] = dt == data_type_t::bf16
                ? static_cast<float>(static_cast<const bfloat16_t *>(buff)[i])
                : static_cast<float>(static_cast<const float16_t *>(buff)[i]);
    }
    return buf.data();
}

/// dst[m * ldd + k] = element (m, k) of an M x K matrix stored row-major
/// (lds apart), or column-major when trans; columns K .. ldd are zeroed.
/// Blocks of block rows (by block columns when trans), on nt threads when
/// M * K >= par_min.  Out of line: its loop would otherwise widen the frame
/// of quantize_source, which every dynamically quantized call enters.
template <typename T>
ZENDNNL_NOINLINE void copy_rows(const void *src, int lds, bool trans, int M,
        int K, void *dst, int ldd, int nt, int block, int64_t par_min) {
    const T *s = static_cast<const T *>(src);
    T *d = static_cast<T *>(dst);
    const size_t ls = static_cast<size_t>(lds);
    const size_t ld = static_cast<size_t>(ldd);
    const int m_blocks = ceil_div(M, block);
    const bool parallel = static_cast<int64_t>(M) * K >= par_min;
    native_for(m_blocks, nt, parallel, [=](int mb) {
        // A block ends at its start plus what is left, at most block: the
        // start plus block may pass the int bound.
        const int m0 = mb * block;
        const int m_end = m0 + std::min(block, M - m0);
        for (int m = m0; m < m_end; ++m)
            std::fill(d + m * ld + K, d + (m + 1) * ld, T(0));
        if (!trans) {
            for (int m = m0; m < m_end; ++m)
                std::memcpy(d + m * ld, s + m * ls, sizeof(T) * K);
            return;
        }
        for (int k0 = 0; k0 < K;) {
            const int k_end = k0 + std::min(block, K - k0);
            for (int k = k0; k < k_end; ++k)
                for (int m = m0; m < m_end; ++m)
                    d[m * ld + k] = s[k * ls + m];
            k0 = k_end;
        }
    });
}

/// shift[g] = -(1 << bits) times the sum of the s8 values of group g of one
/// row: groups groups of group_bytes bytes (a multiple of 4) from row.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vnni")
void row_group_shifts(const uint8_t *row, int groups, int group_bytes, int bits,
        int32_t *shift) {
    const __m512i ones = _mm512_set1_epi8(1);
    const int quads = group_bytes / 4;
    if (group_bytes < 64 && 16 % quads == 0) {
        // The quad sums of 64 bytes, added up over the lanes of each group
        // they hold.
        const int64_t bytes = static_cast<int64_t>(groups) * group_bytes;
        alignas(64) int32_t lanes[16];
        int g = 0;
        for (int64_t k = 0; k < bytes; k += 64) {
            const int64_t left = bytes - k;
            const __mmask64 in
                    = _cvtu64_mask64(left >= 64 ? ~0ull : ~0ull >> (64 - left));
            __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones,
                    _mm512_maskz_loadu_epi8(in, row + k));
            if (quads >= 2)
                s = _mm512_add_epi32(s, _mm512_shuffle_epi32(s, _MM_PERM_CDAB));
            if (quads >= 4)
                s = _mm512_add_epi32(s, _mm512_shuffle_epi32(s, _MM_PERM_BADC));
            if (quads >= 8)
                s = _mm512_add_epi32(
                        s, _mm512_shuffle_i32x4(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
            _mm512_store_si512(lanes,
                    _mm512_sub_epi32(_mm512_setzero_si512(),
                            _mm512_sll_epi32(s, _mm_cvtsi32_si128(bits))));
            const int n = std::min(16 / quads, groups - g);
            for (int j = 0; j < n; ++j)
                shift[g + j] = lanes[j * quads];
            g += n;
        }
        return;
    }
    for (int g = 0; g < groups; ++g) {
        const uint8_t *p = row + static_cast<size_t>(g) * group_bytes;
        __m512i acc = _mm512_setzero_si512();
        int k = 0;
        for (; k + 64 <= group_bytes; k += 64)
            acc = _mm512_dpbusd_epi32(acc, ones, _mm512_loadu_si512(p + k));
        if (k < group_bytes)
            acc = _mm512_dpbusd_epi32(acc, ones,
                    _mm512_maskz_loadu_epi8(
                            _cvtu64_mask64(~0ull >> (64 - (group_bytes - k))),
                            p + k));
        shift[g] = _mm512_reduce_add_epi32(acc) * -(1 << bits);
    }
}

/// Dynamic quantization of a bf16 / f32 source by matmul's per-call reorder
/// quantization into call.src_buffers.  On success q describes the quantized
/// source (reorder_quantize_source), and trans says whether its rows are
/// still transposed.  Out of line: it would otherwise sit in
/// prepare_int8_call, which every INT8 call enters.
ZENDNNL_NOINLINE status_t quantize_source(const GemmDescriptor &desc,
        const matmul_params &params, const void *src, int nt, int copy_block,
        int64_t copy_par_min, int groups, native_int8_call_t &call,
        const void *&a, int &lda, bool &trans, const matmul_params *&q,
        const char **why) {
    const int M = desc.M;
    const int K = desc.K;
    const auto &qp = params.quant_params;
    const data_type_t compute = params.dtypes.compute;
    if (!params.dynamic_quant
            || (compute != data_type_t::s8 && compute != data_type_t::u8))
        return turned_away(status_t::unimplemented,
                "a bf16 / f32 source with an s8 weight and no dynamic "
                "quantization to u8 / s8",
                why);
    const int64_t n_scale
            = quant_param_tensor_or_row_values(qp.src_scale.dims, M);
    const bool per_group_src
            = quant_param_is_per_group_src(qp.src_scale, M, groups);
    if (n_scale == 0 && !per_group_src)
        return turned_away(status_t::unimplemented,
                "a dynamic quantization with a source scale other than per "
                "tensor, per token or per group",
                why);
    // The quantizer writes one zero point per scale.
    if (compute == data_type_t::u8
            && quant_param_tensor_or_row_values(qp.src_zp.dims, M) != n_scale)
        return turned_away(status_t::unimplemented,
                "a dynamic quantization to u8 with a zero point of another "
                "shape than its scale",
                why);

    // A transposed source is quantized as given, as matmul's other routes
    // quantize it, with its quantized rows still transposed, where the
    // quantization of the same rows made row-major gives other values:
    // with a weight whose format's other route quantizes it so (s4,
    // native_contract_t::transposed_as_given), and quantized to s8 with a
    // scale per row (per tensor at M 1) or per group stored narrower than
    // f32, which the row kernels quantize with before they narrow it.  Any
    // other transposed source is made row-major first (a quantization to u8
    // has to be: the shared quantizer takes no per-row zero points of a
    // transposed source).
    const bool narrowed_rows = qp.src_scale.dt != data_type_t::f32
            && (n_scale == M || per_group_src);
    trans = desc.transA
            && (native_format(desc.wei_dt).contract.transposed_as_given
                    || (compute == data_type_t::s8 && narrowed_rows));
    const void *rows = src;
    int rows_ld = desc.lda;
    std::vector<uint8_t> transposed;
    if (desc.transA && !trans) {
        const size_t es = size_of(params.dtypes.src);
        transposed.resize(static_cast<size_t>(M) * K * es);
        if (es == sizeof(uint16_t))
            copy_rows<uint16_t>(src, desc.lda, true, M, K, transposed.data(), K,
                    nt, copy_block, copy_par_min);
        else
            copy_rows<uint32_t>(src, desc.lda, true, M, K, transposed.data(), K,
                    nt, copy_block, copy_par_min);
        rows = transposed.data();
        rows_ld = K;
    }

    const void *quantized = rows;
    int quantized_lda = rows_ld;
    status_t st = status_t::success;
    const auto quantize = [&]() {
        st = reorder_quantize_source(params, rows, rows_ld, M, K, nt,
                call.src_buffers, quantized, quantized_lda, q, trans);
    };
    // Fewer source groups than the team are quantized by the calling thread
    // in a region of the whole team: a region of another size than the
    // team's resizes it, twice per call.
    if (n_scale == 0 && !omp_in_parallel()
            && static_cast<int64_t>(M) * groups < nt) {
        std::exception_ptr error;
#pragma omp parallel num_threads(nt)
        {
            if (omp_get_thread_num() == 0) {
                try {
                    quantize();
                } catch (...) { error = std::current_exception(); }
            }
        }
        if (error) std::rethrow_exception(error);
    } else {
        quantize();
    }
    if (st != status_t::success) return st;
    a = quantized;
    lda = quantized_lda;
    // The wrapper leaves the source as it was when it does not quantize.
    if (q->dtypes.src != data_type_t::s8 && q->dtypes.src != data_type_t::u8)
        return turned_away(status_t::unimplemented,
                "a dynamic quantization the quantizer does not make", why);
    return status_t::success;
}

/// The types of a zero point of a u8 / s8 source.
bool zero_point_dt(data_type_t dt) {
    return dt == data_type_t::s32 || dt == data_type_t::s8
            || dt == data_type_t::u8;
}

/// The destinations of a per-group call: f32 or bf16, or u8 / s8 / s32,
/// which the epilogue requantizes and rounds as the per-channel call's.
bool group_dst(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::u8 || dt == data_type_t::s8
            || dt == data_type_t::s32;
}

/// check_int8_call of a call with groups per-group weight scales on a weight
/// of format f.
status_t check_group_call(const native_format_desc_t &f,
        const GemmDescriptor &desc, const matmul_params &params, int groups,
        const char **why) {
    const auto &qp = params.quant_params;
    const quant_t &sc = qp.src_scale;
    if (!quant_groups_are_whole_quads(desc.K, groups))
        return turned_away(status_t::unimplemented,
                "per-group weight scales over groups of other than whole K "
                "quads",
                why);
    if (!native_takes_scales(f, qp.wei_scale.dt)
            || (sc.buff != nullptr && !native_float_dt(sc.dt)))
        return turned_away(status_t::unimplemented,
                "a per-group call with scales other than f32, bf16 or f16",
                why);
    if (!group_dst(desc.dst_dt))
        return turned_away(status_t::unimplemented,
                "a per-group call into a dst other than f32, bf16, u8, s8 or "
                "s32",
                why);
    const int64_t n_scale = quant_param_tensor_or_row_values(sc.dims, desc.M);
    const bool per_group = quant_param_is_per_group_src(sc, desc.M, groups);
    // A u8 source, given or quantized in the call, has one scale and one
    // zero point for the whole tensor.
    const bool tensor_zp
            = quant_param_tensor_or_row_values(qp.src_zp.dims, desc.M) == 1;
    if (dynamic_source(params)) {
        if (params.dtypes.compute == data_type_t::u8)
            return n_scale == 1 && tensor_zp
                            && (qp.src_zp.buff == nullptr
                                    || qp.src_zp.dt == data_type_t::s32)
                    ? status_t::success
                    : turned_away(status_t::unimplemented,
                              "a per-group call quantizing its source to u8 "
                              "with other than one scale and one s32 zero "
                              "point",
                              why);
        return params.dtypes.compute == data_type_t::s8
                        && (n_scale == 1 || quant_param_is_per_token(sc, desc.M)
                                || per_group)
                ? status_t::success
                : turned_away(status_t::unimplemented,
                          "a per-group call quantizing its source with a scale "
                          "other than per tensor, per token or per group",
                          why);
    }
    if (params.dtypes.src == data_type_t::u8)
        return (sc.buff == nullptr || n_scale == 1)
                        && (qp.src_zp.buff == nullptr
                                || (tensor_zp && zero_point_dt(qp.src_zp.dt)))
                ? status_t::success
                : turned_away(status_t::unimplemented,
                          "a per-group call with a u8 source whose scale or "
                          "zero "
                          "point is not per tensor",
                          why);
    if (params.dtypes.src != data_type_t::s8)
        return turned_away(status_t::unimplemented,
                "a per-group call with a source other than u8 or s8", why);
    if (qp.src_zp.buff != nullptr)
        return turned_away(status_t::unimplemented,
                "an s8 source with a zero point on per-group weights", why);
    if (sc.buff != nullptr && n_scale == 0 && !per_group)
        return turned_away(status_t::unimplemented,
                "a per-group call with a source scale other than per tensor, "
                "per token or per group",
                why);
    return status_t::success;
}

/// check_int8_call of a call with an {M, groups} source scale and
/// per-tensor or per-channel weight scales (or none) on a weight of format
/// f: an s8 source without a zero point, given or quantized in the call.
status_t check_source_group_call(const native_format_desc_t &f,
        const GemmDescriptor &desc, const matmul_params &params, int groups,
        const char **why) {
    const auto &qp = params.quant_params;
    const quant_t &ws = qp.wei_scale;
    if (!quant_groups_are_whole_quads(desc.K, groups) || !group_dst(desc.dst_dt)
            || !native_float_dt(qp.src_scale.dt) || qp.src_zp.buff != nullptr
            || (ws.buff != nullptr
                    && ((!quant_param_dims_are_per_tensor(ws.dims)
                                && !quant_param_is_per_channel(ws, desc.N))
                            || !native_takes_scales(f, ws.dt))))
        return turned_away(status_t::unimplemented,
                "an {M, G} source scale with a zero point, a dst, groups or "
                "weight scales the per-group tile does not take",
                why);
    return int8_s8_source(params)
            ? status_t::success
            : turned_away(status_t::unimplemented,
                      "an {M, G} source scale with a source other than s8",
                      why);
}

} // namespace

native_int8_decision_t decide_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias) {
    native_int8_decision_t d;
    const data_type_t dst = desc.dst_dt;
    if ((native_type_bit(desc.wei_dt) & kNativeGroupedTypes) != 0) {
        d.scheme = native_int8_scheme_t::per_group;
        d.groups = int4_scale_groups(params, desc.N);
        d.direct_store = desc.alpha == 1.0f && desc.beta == 0.0f
                && (dst == data_type_t::f32 || dst == data_type_t::bf16);
        d.order = d.direct_store ? native_int8_order_t::fused
                                 : native_int8_order_t::reference;
        // The reference algorithm applies alpha and beta to the sum of one
        // group over K before both scales.
        d.scales_after_groups
                = d.groups == 1 && (desc.alpha != 1.0f || desc.beta != 0.0f);
        return d;
    }
    d.groups = int8_scale_groups(params, desc.N);
    if (d.groups > 0) {
        d.scheme = native_int8_scheme_t::per_group;
        d.order = native_int8_order_t::reference;
        d.direct_store = desc.alpha == 1.0f && desc.beta == 0.0f
                && (dst == data_type_t::f32 || dst == data_type_t::bf16);
        return d;
    }
    const int source_groups = int8_source_groups(params, desc.M, desc.N);
    if (source_groups > 0) {
        d.scheme = native_int8_scheme_t::per_group;
        d.groups = source_groups;
        d.scales_after_groups = true;
        d.order = native_int8_order_t::reference;
        return d;
    }
    if (desc.alpha != 1.0f || desc.beta != 0.0f || dst == data_type_t::s8
            || dst == data_type_t::u8 || dst == data_type_t::s32) {
        d.order = native_int8_order_t::reference;
        return d;
    }
    const data_type_t bias_dt = params.dtypes.bias;
    d.bias_in_dequant = bias != nullptr
            && (bias_dt == data_type_t::f32 || bias_dt == data_type_t::bf16
                    || bias_dt == data_type_t::f16);
    d.direct_store = dst == data_type_t::f32 || dst == data_type_t::bf16;
    return d;
}

status_t check_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        const char **why) {
    const int M = desc.M;
    const native_format_desc_t &f = native_format(params.dtypes.wei);
    const native_contract_t &contract = f.contract;
    if (desc.K > contract.k.max)
        return turned_away(status_t::unimplemented, contract.k.why, why);
    const auto &qp = params.quant_params;
    if (qp.wei_zp.buff != nullptr)
        return turned_away(
                status_t::unimplemented, contract.wei_zp_refusal, why);
    // The tile applies beta of a gated call over the dst's N pre-activation
    // columns.
    if (desc.gated_act != matmul_gated_act_t::none && desc.beta != 0.0f
            && desc.ldc < desc.N)
        return turned_away(status_t::unimplemented,
                "beta with a gated activation over a dst of fewer than N "
                "columns",
                why);
    if (contract.layout_k_refusal != nullptr && desc.K > native_max_layout_k(f))
        return turned_away(
                status_t::unimplemented, contract.layout_k_refusal, why);
    // An integer source the kernels would take (the one the call quantizes
    // to, or the call's) that the format refuses; the quantization checks
    // below refuse the others.
    const data_type_t int_src = dynamic_source(params) ? params.dtypes.compute
                                                       : params.dtypes.src;
    if (int_src == data_type_t::u8 || int_src == data_type_t::s8)
        for (const native_refusal_t &r : contract.src_refusals)
            if ((r.types & native_type_bit(int_src)) != 0)
                return turned_away(status_t::unimplemented, r.why, why);
    // A source zero point, also on a source the call quantizes, whose
    // symmetric quantization would drop it.
    if (contract.src_zp_refusal != nullptr && qp.src_zp.buff != nullptr)
        return turned_away(
                status_t::unimplemented, contract.src_zp_refusal, why);
    if (contract.scale_grains == native_grain_bit(native_grain_t::group)) {
        if (decision.groups == 0)
            return turned_away(
                    status_t::unimplemented, contract.ungrouped_refusal, why);
        if (desc.K / decision.groups > contract.group.max)
            return turned_away(
                    status_t::unimplemented, contract.group.why, why);
        return check_group_call(f, desc, params, decision.groups, why);
    }
    // The symmetric quantization of a source the call quantizes to s8 would
    // drop a source zero point.
    if (dynamic_source(params) && params.dtypes.compute == data_type_t::s8
            && qp.src_zp.buff != nullptr)
        return turned_away(status_t::unimplemented,
                "a source the call quantizes to s8 with a source zero point",
                why);
    if (decision.scheme == native_int8_scheme_t::per_group)
        return decision.scales_after_groups
                ? check_source_group_call(f, desc, params, decision.groups, why)
                : check_group_call(f, desc, params, decision.groups, why);

    const data_type_t src_dt = params.dtypes.src;
    if (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32) {
        const data_type_t compute = params.dtypes.compute;
        if (!params.dynamic_quant
                || (compute != data_type_t::s8 && compute != data_type_t::u8))
            return turned_away(status_t::unimplemented,
                    "a bf16 / f32 source with an s8 weight and no dynamic "
                    "quantization to u8 / s8",
                    why);
        // What the shared quantizer takes: a per-tensor or an {M, 1}
        // per-token scale of f32 / bf16 / f16, and for u8 an s32 zero point
        // of the scale's shape.
        const auto &sc = qp.src_scale;
        const int64_t n_scale = quant_param_tensor_or_row_values(sc.dims, M);
        if (n_scale == 0 || (n_scale != 1 && !quant_param_is_per_token(sc, M))
                || (sc.buff != nullptr && !native_float_dt(sc.dt)))
            return turned_away(status_t::unimplemented,
                    "a dynamic quantization with a source scale other than "
                    "per tensor or per token of f32, bf16 or f16",
                    why);
        if (compute == data_type_t::u8
                && (quant_param_tensor_or_row_values(qp.src_zp.dims, M)
                                != n_scale
                        || (qp.src_zp.buff != nullptr
                                && qp.src_zp.dt != data_type_t::s32)))
            return turned_away(status_t::unimplemented,
                    "a dynamic quantization to u8 with a zero point other "
                    "than s32 of its scale's shape",
                    why);
    } else if (src_dt == data_type_t::u8 || src_dt == data_type_t::s8) {
        if (qp.src_scale.buff != nullptr
                && (quant_param_tensor_or_row_values(qp.src_scale.dims, M) == 0
                        || !native_float_dt(qp.src_scale.dt)))
            return turned_away(status_t::unimplemented,
                    "a source scale other than per tensor or per row of f32, "
                    "bf16 or f16",
                    why);
        // A u8 / s8 source keeps its zero point whatever dynamic_quant says.
        if (qp.src_zp.buff != nullptr) {
            const int64_t n_zp
                    = quant_param_tensor_or_row_values(qp.src_zp.dims, M);
            if (n_zp == 0
                    || (qp.src_zp.dt != data_type_t::s32
                            && qp.src_zp.dt != data_type_t::s8
                            && qp.src_zp.dt != data_type_t::u8))
                return turned_away(status_t::unimplemented,
                        "a source zero point other than per tensor or per "
                        "row of s32, s8 or u8",
                        why);
        }
    } else {
        return turned_away(status_t::unimplemented,
                "a source other than u8, s8, bf16 or f32 with an s8 weight",
                why);
    }

    if (qp.wei_scale.buff != nullptr
            && ((!quant_param_dims_are_per_tensor(qp.wei_scale.dims)
                        && !quant_param_is_per_channel(qp.wei_scale, desc.N))
                    || !native_takes_scales(f, qp.wei_scale.dt)))
        return turned_away(status_t::unimplemented,
                "weight scales other than per tensor, per column or per group "
                "of f32, bf16 or f16",
                why);
    return status_t::success;
}

int int4_scale_groups(const matmul_params &params, int N) {
    const quant_t &ws = params.quant_params.wei_scale;
    if (ws.buff == nullptr || ws.dims.size() != 2 || ws.dims[0] < 1
            || ws.dims[1] != static_cast<int64_t>(N))
        return 0;
    return static_cast<int>(
            std::min<int64_t>(ws.dims[0], std::numeric_limits<int>::max()));
}

void copy_int8_source(const void *src, int lds, bool trans, int M, int K,
        uint8_t *dst, int ldd, int nt, int block, int64_t par_min) {
    copy_rows<uint8_t>(src, lds, trans, M, K, dst, ldd, nt, block, par_min);
}

status_t prepare_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        native_route_t route, const void *src, int copy_block,
        int64_t copy_par_min, native_int8_call_t &call, const char **why) {
    const int M = desc.M;
    const int K = desc.K;
    if (M <= 0 || K <= 0 || src == nullptr || desc.lda < (desc.transA ? M : K))
        return status_t::failure;
    const auto &qp = params.quant_params;
    const int groups = decision.groups;
    const native_format_desc_t &f = native_format(desc.wei_dt);

    assert(native_route(route).format == f.id);
    call.route = route;
    call.A = nullptr;
    call.lda = 0;
    call.src_signed = false;
    call.src_zp = nullptr;
    call.src_zp_stride = 0;
    call.bias = nullptr;
    call.alpha = desc.alpha;
    call.beta = desc.beta;
    call.scheme = decision.scheme;
    call.order = decision.order;
    call.dst = nullptr;
    call.dst_dt = data_type_t::none;
    call.ldc = 0;
    call.wide_zp = false;
    call.groups = 0;
    call.group_quads = 0;
    call.src_scale_group_stride = 0;
    call.group_shift = nullptr;
    call.shift_bits = f.offset_bits;

    const int nt = omp_in_parallel() ? 1 : std::max(1, desc.num_threads);
    const data_type_t src_dt = params.dtypes.src;
    const void *a = src;
    int lda = desc.lda;
    bool trans = desc.transA;
    data_type_t a_dt = src_dt;
    // A symmetric dynamic quantization has no zero point (check_int8_call
    // refuses a buffer of one).
    const bool dynamic_sym = dynamic_source(params)
            && params.dtypes.compute != data_type_t::u8;
    const quant_t *src_scale = &qp.src_scale;
    const quant_t *src_zp = dynamic_sym ? nullptr : &qp.src_zp;
    const matmul_params *quantized = nullptr;
    if (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32) {
        const status_t st = quantize_source(desc, params, src, nt, copy_block,
                copy_par_min, groups, call, a, lda, trans, quantized, why);
        if (st != status_t::success) return st;
        a_dt = quantized->dtypes.src;
        src_scale = &quantized->quant_params.src_scale;
        if (src_zp != nullptr) src_zp = &quantized->quant_params.src_zp;
    } else if (src_dt != data_type_t::u8 && src_dt != data_type_t::s8) {
        return turned_away(status_t::unimplemented,
                "a source other than u8, s8, bf16 or f32 with an s8 weight",
                why);
    }

    call.src_signed = a_dt == data_type_t::s8;
    // The kernels read the source by the K values of the weight's words
    // (octets of s4 nibbles, else quads); a zero source value adds nothing
    // against any padded weight.
    const int k_word = f.geometry.k_per_word;
    const int k_pad = ceil_div(K, k_word) * k_word;
    if (trans || k_pad != K) {
        call.a_buf.resize(static_cast<size_t>(M) * k_pad);
        copy_int8_source(a, lda, trans, M, K, call.a_buf.data(), k_pad, nt,
                copy_block, copy_par_min);
        call.A = call.a_buf.data();
        call.lda = k_pad;
    } else {
        call.A = static_cast<const uint8_t *>(a);
        call.lda = lda;
    }

    if (src_scale->buff == nullptr) {
        call.src_scale = &kUnitScale;
        call.src_scale_stride = 0;
    } else if (quant_param_is_per_group_src(*src_scale, M, groups)) {
        if (!native_float_dt(src_scale->dt))
            return turned_away(status_t::unimplemented,
                    "a source scale other than f32, bf16 or f16", why);
        call.src_scale = f32_values(src_scale->buff, src_scale->dt,
                static_cast<int64_t>(M) * groups, call.scale_buf);
        call.src_scale_stride = groups;
        call.src_scale_group_stride = 1;
    } else {
        const int64_t n = quant_param_tensor_or_row_values(src_scale->dims, M);
        if (n == 0 || !native_float_dt(src_scale->dt))
            return turned_away(status_t::unimplemented,
                    "a source scale other than per tensor or per row of f32, "
                    "bf16 or f16",
                    why);
        call.src_scale
                = f32_values(src_scale->buff, src_scale->dt, n, call.scale_buf);
        call.src_scale_stride = n == 1 ? 0 : 1;
    }

    if (src_zp != nullptr && src_zp->buff != nullptr) {
        const int64_t n = quant_param_tensor_or_row_values(src_zp->dims, M);
        if (n == 0)
            return turned_away(status_t::unimplemented,
                    "a source zero point other than per tensor or per row",
                    why);
        if (src_zp->dt == data_type_t::s32) {
            call.src_zp = static_cast<const int32_t *>(src_zp->buff);
        } else if (src_zp->dt == data_type_t::s8
                || src_zp->dt == data_type_t::u8) {
            call.zp_buf.resize(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                call.zp_buf[i] = src_zp->dt == data_type_t::s8
                        ? static_cast<const int8_t *>(src_zp->buff)[i]
                        : static_cast<const uint8_t *>(src_zp->buff)[i];
            }
            call.src_zp = call.zp_buf.data();
        } else {
            return turned_away(status_t::unimplemented,
                    "a source zero point other than s32, s8 or u8", why);
        }
        call.src_zp_stride = n == 1 ? 0 : 1;
        // An asymmetric quantization puts the zero point of a row whose
        // values exclude 0 outside the byte range, and so can a caller.  The
        // group kernel takes one zero point.  The int32 sums over K, or over
        // a group, are exact only within the int32 range: past it the
        // kernels take the zero-point term in int64.
        const bool per_group
                = decision.scheme == native_int8_scheme_t::per_group;
        if (per_group && n != 1)
            return turned_away(status_t::unimplemented,
                    "a per-group call whose source zero point is not one value",
                    why);
        call.wide_zp = !zero_points_fit(
                call.src_zp, n, call.src_signed, per_group ? K / groups : K);
    }

    const auto &wei_scale = qp.wei_scale;
    if (wei_scale.buff == nullptr) {
        call.wei_scale = &kUnitScale;
        call.wei_scale_type = native_scale_t::f32;
        call.wei_scale_stride = 0;
    } else if (decision.scheme == native_int8_scheme_t::per_group
            && !decision.scales_after_groups) {
        call.wei_scale = wei_scale.buff;
        call.wei_scale_type = native_scale_of(wei_scale.dt);
        call.wei_scale_stride = 1;
    } else {
        int64_t n = 0;
        if (quant_param_dims_are_per_tensor(wei_scale.dims))
            n = 1;
        else if (quant_param_is_per_channel(wei_scale, desc.N))
            n = desc.N;
        if (n == 0 || !native_takes_scales(f, wei_scale.dt))
            return turned_away(status_t::unimplemented,
                    "weight scales other than per tensor or per column of "
                    "f32, bf16 or f16",
                    why);
        call.wei_scale = wei_scale.buff;
        call.wei_scale_type = native_scale_of(wei_scale.dt);
        call.wei_scale_stride = n == 1 ? 0 : 1;
    }
    assert((f.contract.scale_types & native_scale_bit(call.wei_scale_type))
            != 0);
    call.scales_after_groups = decision.scales_after_groups;
    if (call.scheme == native_int8_scheme_t::per_group) {
        call.groups = groups;
        call.group_quads = K / groups / 4;
    }
    if (call.scheme == native_int8_scheme_t::per_group && call.src_signed) {
        // Every shift is written before its chunk is marked ready, and read
        // only after: the buffer needs no fill.
        native_int8_shift_store_t *const shift_store = call.shift_store;
        const size_t shifts = static_cast<size_t>(M) * groups;
        if (shift_store != nullptr
                && shifts <= native_int8_shift_store_t::kShifts) {
            call.group_shift = shift_store->shifts;
        } else {
            call.shift_buf.reset(new int32_t[shifts]);
            call.group_shift = call.shift_buf.get();
        }
        call.rows = M;
        call.shift_chunk = int8_shift_claim_rows(K);
        const int chunks = ceil_div(M, call.shift_chunk);
        if (shift_store != nullptr
                && chunks <= native_int8_shift_store_t::kChunks) {
            call.shift_rows = shift_store->rows;
        } else {
            call.shift_rows_buf.reset(new std::atomic<int>[chunks + 1]);
            call.shift_rows = call.shift_rows_buf.get();
        }
        for (int i = 0; i <= chunks; ++i)
            call.shift_rows[i].store(0, std::memory_order_relaxed);
    }
    return status_t::success;
}

void ready_group_shifts(const native_int8_call_t &call, int m0, int mr) {
    std::atomic<int> *const next = &call.shift_rows[0];
    std::atomic<int> *const ready = &call.shift_rows[1];
    const int chunk = call.shift_chunk;
    const int chunks = ceil_div(call.rows, chunk);
    const int first = m0 / chunk;
    const int end = ceil_div(m0 + mr, chunk);
    const int group_bytes = call.group_quads * 4;
    // Chunks are claimed in order: once next reaches end, every chunk of the
    // range is claimed.
    while (next->load(std::memory_order_relaxed) < end) {
        const int c = next->fetch_add(1, std::memory_order_relaxed);
        if (c >= chunks) break;
        const int m_end = c * chunk + std::min(chunk, call.rows - c * chunk);
        for (int m = c * chunk; m < m_end; ++m)
            row_group_shifts(call.A + static_cast<size_t>(m) * call.lda,
                    call.groups, group_bytes, call.shift_bits,
                    call.group_shift + static_cast<size_t>(m) * call.groups);
        ready[c].store(1, std::memory_order_release);
    }
    // One budget of pauses for the whole wait, then yields.
    int spin = 0;
    for (int c = first; c < end; ++c) {
        while (ready[c].load(std::memory_order_acquire) == 0) {
            if (spin < kShiftSpinPauses) {
                ++spin;
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
