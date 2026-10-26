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
#include <cstddef>
#include <cstring>
#include <limits>

#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"

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

/// 1 for a per-tensor buffer, rows for a per-row one ({rows} or {rows, 1}),
/// 0 for any other shape.
int64_t per_tensor_or_row(const std::vector<int64_t> &dims, int rows) {
    const int64_t n = checked_quant_param_num_elements(dims);
    if (n == 1) return 1;
    const bool row_shape
            = dims.size() == 1 || (dims.size() == 2 && dims[1] == 1);
    return n == rows && row_shape ? n : 0;
}

/// An {M, groups} source scale: one per row and K group.
bool per_group_scale(const quant_t &q, int M, int groups) {
    return groups > 1 && q.dims.size() == 2 && q.dims[0] == M
            && q.dims[1] == groups;
}

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

/// A dynamic quantization of the source: bf16 / f32 quantized in the call.
bool dynamic_source(const matmul_params &params) {
    return params.dynamic_quant
            && (params.dtypes.src == data_type_t::bf16
                    || params.dtypes.src == data_type_t::f32);
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
/// M * K >= par_min.
template <typename T>
void copy_rows(const void *src, int lds, bool trans, int M, int K, void *dst,
        int ldd, int nt, int block, int64_t par_min) {
    const T *s = static_cast<const T *>(src);
    T *d = static_cast<T *>(dst);
    const size_t ls = static_cast<size_t>(lds);
    const size_t ld = static_cast<size_t>(ldd);
    const int m_blocks = (M + block - 1) / block;
    const bool parallel = nt > 1 && static_cast<int64_t>(M) * K >= par_min;
#pragma omp parallel for num_threads(nt) if (parallel) schedule(static)
    for (int mb = 0; mb < m_blocks; ++mb) {
        const int m_end = std::min(M, (mb + 1) * block);
        for (int m = mb * block; m < m_end; ++m)
            std::fill(d + m * ld + K, d + (m + 1) * ld, T(0));
        if (!trans) {
            for (int m = mb * block; m < m_end; ++m)
                std::memcpy(d + m * ld, s + m * ls, sizeof(T) * K);
            continue;
        }
        for (int k0 = 0; k0 < K; k0 += block) {
            const int k_end = std::min(K, k0 + block);
            for (int k = k0; k < k_end; ++k)
                for (int m = mb * block; m < m_end; ++m)
                    d[m * ld + k] = s[k * ls + m];
        }
    }
}

/// Dynamic quantization of a bf16 / f32 source by matmul's per-call reorder
/// quantization into call.src_buffers.  On success q describes the quantized
/// source.
status_t quantize_source(const GemmDescriptor &desc,
        const matmul_params &params, const void *src, int nt, int copy_block,
        int64_t copy_par_min, int groups, native_int8_call_t &call,
        const void *&a, int &lda, matmul_params &q) {
    const int M = desc.M;
    const int K = desc.K;
    const auto &qp = params.quant_params;
    const data_type_t compute = params.dtypes.compute;
    if (!params.dynamic_quant
            || (compute != data_type_t::s8 && compute != data_type_t::u8))
        return status_t::unimplemented;
    const int64_t n_scale = per_tensor_or_row(qp.src_scale.dims, M);
    if (n_scale == 0 && !per_group_scale(qp.src_scale, M, groups))
        return status_t::unimplemented;
    // The quantizer writes one zero point per scale.
    if (compute == data_type_t::u8
            && per_tensor_or_row(qp.src_zp.dims, M) != n_scale)
        return status_t::unimplemented;

    // A transposed source is made row-major first, so it quantizes exactly
    // as the same rows given row-major.
    const void *rows = src;
    int rows_ld = desc.lda;
    std::vector<uint8_t> transposed;
    if (desc.transA) {
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

    // What the quantizer reads, and rewrites for the quantized source.
    q.dtypes = params.dtypes;
    q.quant_params = params.quant_params;
    q.dynamic_quant = params.dynamic_quant;
    q.packing = params.packing;
    const void *quantized = rows;
    int quantized_lda = rows_ld;
    size_t elem_size = size_of(params.dtypes.src);
    matmul_batch_params_t batch;
    // The quantization kernels size their thread teams from the nthreads ICV.
    const thread_guard team(nt);
    const status_t st
            = reorder_quantization_wrapper(quantized, rows_ld, quantized_lda,
                    elem_size, q, batch, false, M, K, nt, call.src_buffers);
    if (st != status_t::success) return st;
    a = quantized;
    lda = quantized_lda;
    // The wrapper leaves the source as it was when it does not quantize;
    // another route can still serve the call.
    if (q.dtypes.src != data_type_t::s8 && q.dtypes.src != data_type_t::u8)
        return status_t::unimplemented;
    return status_t::success;
}

/// check_int8_call of a call with groups per-group weight scales.
status_t check_group_call(
        const GemmDescriptor &desc, const matmul_params &params, int groups) {
    const auto &qp = params.quant_params;
    const quant_t &sc = qp.src_scale;
    if (desc.K % groups != 0 || (desc.K / groups) % 4 != 0
            || !native_float_dt(qp.wei_scale.dt)
            || (desc.dst_dt != data_type_t::f32
                    && desc.dst_dt != data_type_t::bf16)
            || (sc.buff != nullptr && !native_float_dt(sc.dt)))
        return status_t::unimplemented;
    const int64_t n_scale = per_tensor_or_row(sc.dims, desc.M);
    const bool per_group = per_group_scale(sc, desc.M, groups);
    if (dynamic_source(params))
        return params.dtypes.compute == data_type_t::s8
                        && (n_scale == 1 || quant_param_is_per_token(sc, desc.M)
                                || per_group)
                ? status_t::success
                : status_t::unimplemented;
    if (params.dtypes.src != data_type_t::s8 || qp.src_zp.buff != nullptr
            || (sc.buff != nullptr && n_scale == 0 && !per_group))
        return status_t::unimplemented;
    return status_t::success;
}

} // namespace

native_int8_decision_t decide_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias) {
    native_int8_decision_t d;
    d.groups = int8_scale_groups(params, desc.N);
    if (d.groups > 0) {
        d.scheme = native_int8_scheme_t::per_group;
        d.order = native_int8_order_t::reference;
        return d;
    }
    const data_type_t dst = desc.dst_dt;
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
        const matmul_params &params, const native_int8_decision_t &decision) {
    const int M = desc.M;
    if (params.dtypes.wei != data_type_t::s8 || desc.K > kMaxSafeInt8AccumK)
        return status_t::unimplemented;
    const auto &qp = params.quant_params;
    if (qp.wei_zp.buff != nullptr) return status_t::unimplemented;
    if (desc.gated_act != matmul_gated_act_t::none && desc.beta != 0.0f)
        return status_t::unimplemented;
    if (decision.scheme == native_int8_scheme_t::per_group)
        return check_group_call(desc, params, decision.groups);

    const data_type_t src_dt = params.dtypes.src;
    if (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32) {
        const data_type_t compute = params.dtypes.compute;
        if (!params.dynamic_quant
                || (compute != data_type_t::s8 && compute != data_type_t::u8))
            return status_t::unimplemented;
        // What the shared quantizer takes: a per-tensor or an {M, 1}
        // per-token scale of f32 / bf16 / f16, and for u8 an s32 zero point
        // of the scale's shape.
        const auto &sc = qp.src_scale;
        const int64_t n_scale = per_tensor_or_row(sc.dims, M);
        if (n_scale == 0 || (n_scale != 1 && !quant_param_is_per_token(sc, M))
                || (sc.buff != nullptr && !native_float_dt(sc.dt)))
            return status_t::unimplemented;
        if (compute == data_type_t::u8
                && (per_tensor_or_row(qp.src_zp.dims, M) != n_scale
                        || (qp.src_zp.buff != nullptr
                                && qp.src_zp.dt != data_type_t::s32)))
            return status_t::unimplemented;
    } else if (src_dt == data_type_t::u8 || src_dt == data_type_t::s8) {
        if (qp.src_scale.buff != nullptr
                && (per_tensor_or_row(qp.src_scale.dims, M) == 0
                        || !native_float_dt(qp.src_scale.dt)))
            return status_t::unimplemented;
        // A u8 / s8 source keeps its zero point whatever dynamic_quant says.
        if (qp.src_zp.buff != nullptr) {
            const int64_t n_zp = per_tensor_or_row(qp.src_zp.dims, M);
            if (n_zp == 0
                    || (qp.src_zp.dt != data_type_t::s32
                            && qp.src_zp.dt != data_type_t::s8
                            && qp.src_zp.dt != data_type_t::u8))
                return status_t::unimplemented;
        }
    } else {
        return status_t::unimplemented;
    }

    if (qp.wei_scale.buff != nullptr
            && ((!quant_param_dims_are_per_tensor(qp.wei_scale.dims)
                        && !quant_param_is_per_channel(qp.wei_scale, desc.N))
                    || !native_float_dt(qp.wei_scale.dt)))
        return status_t::unimplemented;
    return status_t::success;
}

int int8_scale_groups(const matmul_params &params, int N) {
    if (!has_per_group_wei_scale(params.quant_params, N)) return 0;
    return static_cast<int>(
            std::min<int64_t>(params.quant_params.wei_scale.dims[0],
                    std::numeric_limits<int>::max()));
}

status_t prepare_int8_call(const GemmDescriptor &desc,
        const matmul_params &params, const native_int8_decision_t &decision,
        const void *src, int copy_block, int64_t copy_par_min,
        native_int8_call_t &call) {
    const int M = desc.M;
    const int K = desc.K;
    if (M <= 0 || K <= 0 || src == nullptr || desc.lda < (desc.transA ? M : K))
        return status_t::failure;
    const auto &qp = params.quant_params;
    const int groups = decision.groups;

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

    const int nt = omp_in_parallel() ? 1 : std::max(1, desc.num_threads);
    const data_type_t src_dt = params.dtypes.src;
    const void *a = src;
    int lda = desc.lda;
    bool trans = desc.transA;
    data_type_t a_dt = src_dt;
    // A symmetric dynamic quantization has no zero point, whatever buffer
    // was passed.
    const bool dynamic_sym = dynamic_source(params)
            && params.dtypes.compute != data_type_t::u8;
    const quant_t *src_scale = &qp.src_scale;
    const quant_t *src_zp = dynamic_sym ? nullptr : &qp.src_zp;
    matmul_params quantized;
    if (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32) {
        const status_t st = quantize_source(desc, params, src, nt, copy_block,
                copy_par_min, groups, call, a, lda, quantized);
        if (st != status_t::success) return st;
        trans = false;
        a_dt = quantized.dtypes.src;
        src_scale = &quantized.quant_params.src_scale;
        if (src_zp != nullptr) src_zp = &quantized.quant_params.src_zp;
    } else if (src_dt != data_type_t::u8 && src_dt != data_type_t::s8) {
        return status_t::unimplemented;
    }

    call.src_signed = a_dt == data_type_t::s8;
    const int k_pad = (K + 3) & ~3;
    if (trans || k_pad != K) {
        call.a_buf.resize(static_cast<size_t>(M) * k_pad);
        copy_rows<uint8_t>(a, lda, trans, M, K, call.a_buf.data(), k_pad, nt,
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
    } else if (per_group_scale(*src_scale, M, groups)) {
        if (!native_float_dt(src_scale->dt)) return status_t::unimplemented;
        call.src_scale = f32_values(src_scale->buff, src_scale->dt,
                static_cast<int64_t>(M) * groups, call.scale_buf);
        call.src_scale_stride = groups;
        call.src_scale_group_stride = 1;
    } else {
        const int64_t n = per_tensor_or_row(src_scale->dims, M);
        if (n == 0 || !native_float_dt(src_scale->dt))
            return status_t::unimplemented;
        call.src_scale
                = f32_values(src_scale->buff, src_scale->dt, n, call.scale_buf);
        call.src_scale_stride = n == 1 ? 0 : 1;
    }

    if (src_zp != nullptr && src_zp->buff != nullptr) {
        const int64_t n = per_tensor_or_row(src_zp->dims, M);
        if (n == 0) return status_t::unimplemented;
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
            return status_t::unimplemented;
        }
        call.src_zp_stride = n == 1 ? 0 : 1;
        // An asymmetric quantization puts the zero point of a row whose
        // values exclude 0 outside the byte range, and so can a caller.
        call.wide_zp = !zero_points_fit(call.src_zp, n, call.src_signed, K);
    }

    const auto &wei_scale = qp.wei_scale;
    if (wei_scale.buff == nullptr) {
        call.wei_scale = &kUnitScale;
        call.wei_scale_dt = data_type_t::f32;
        call.wei_scale_stride = 0;
    } else if (decision.scheme == native_int8_scheme_t::per_group) {
        call.wei_scale = wei_scale.buff;
        call.wei_scale_dt = wei_scale.dt;
        call.wei_scale_stride = 1;
        call.groups = groups;
        call.group_quads = K / groups / 4;
    } else {
        int64_t n = 0;
        if (quant_param_dims_are_per_tensor(wei_scale.dims))
            n = 1;
        else if (quant_param_is_per_channel(wei_scale, desc.N))
            n = desc.N;
        if (n == 0 || !native_float_dt(wei_scale.dt))
            return status_t::unimplemented;
        call.wei_scale = wei_scale.buff;
        call.wei_scale_dt = wei_scale.dt;
        call.wei_scale_stride = n == 1 ? 0 : 1;
    }
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
