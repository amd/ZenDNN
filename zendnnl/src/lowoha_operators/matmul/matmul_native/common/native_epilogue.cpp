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

#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <new>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::log_error;
using zendnnl::memory::read_and_cast;

namespace {

constexpr float kSqrt2OverPi = 0.79788458347320556640625f;
constexpr float kGeluTanhCoeff = 0.044715f;

alignas(64) constexpr int32_t kGateLanes[16]
        = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
alignas(64) constexpr int32_t kUpLanes[16]
        = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

bool is_operand_dt(data_type_t dt) {
    return native_float_dt(dt) || dt == data_type_t::s8 || dt == data_type_t::u8
            || dt == data_type_t::s32;
}

float scalar_at(const native_operand_t &o, int64_t i) {
    return o.as_int ? static_cast<float>(read_and_cast<int32_t>(o.buf, o.dt, i))
                    : read_and_cast<float>(o.buf, o.dt, i);
}

/// ld <= 0 means n_out (the leading dimension of a dense matrix).
status_t make_operand(const void *buf, data_type_t dt, int64_t count,
        int64_t ld, int M, int n_out, bool as_int, native_operand_t &o) {
    if (buf == nullptr || count < 1) return status_t::failure;
    if (!is_operand_dt(dt)) return status_t::unimplemented;
    o = native_operand_t {};
    o.buf = buf;
    o.dt = dt;
    o.esz = static_cast<int>(size_of(dt));
    o.as_int = as_int;
    o.count = count;
    if (count == 1) {
        o.kind = native_operand_t::kind_t::scalar;
        o.value = scalar_at(o, 0);
    } else if (count == n_out) {
        o.kind = native_operand_t::kind_t::column;
    } else if (count == static_cast<int64_t>(M) * n_out) {
        o.kind = native_operand_t::kind_t::matrix;
        o.ld = ld > 0 ? ld : n_out;
        if (o.ld < n_out) return status_t::failure;
    } else {
        o.kind = native_operand_t::kind_t::modulo;
    }
    return status_t::success;
}

void *dst_at(const native_epilogue_t &ep, int m, int n) {
    return static_cast<char *>(ep.dst)
            + (static_cast<size_t>(m) * ep.ldc + n) * ep.dst_esz;
}

ZENDNNL_TARGET("avx512f")
inline __mmask16 lane_mask(int lanes) {
    return lanes >= 16 ? static_cast<__mmask16>(0xFFFF)
                       : static_cast<__mmask16>((1u << lanes) - 1u);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline __m512 load_f32(const void *p, data_type_t dt, __mmask16 k) {
    switch (dt) {
        case data_type_t::f32: return _mm512_maskz_loadu_ps(k, p);
        case data_type_t::bf16:
            return avx512_bf16_to_f32(_mm256_maskz_loadu_epi16(k, p));
        case data_type_t::f16:
            return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, p));
        case data_type_t::s8:
            return _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(k, p)));
        case data_type_t::u8:
            return _mm512_cvtepi32_ps(
                    _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(k, p)));
        case data_type_t::s32:
            return _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(k, p));
        default: return _mm512_setzero_ps();
    }
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline __m512 load_operand(const native_operand_t &o, int m, int n, int n_out,
        __mmask16 k, int lanes) {
    int64_t at = n;
    switch (o.kind) {
        case native_operand_t::kind_t::scalar: return _mm512_set1_ps(o.value);
        case native_operand_t::kind_t::column: break;
        case native_operand_t::kind_t::matrix:
            at = static_cast<int64_t>(m) * o.ld + n;
            break;
        case native_operand_t::kind_t::modulo: {
            alignas(64) float v[16] = {};
            const int64_t base = static_cast<int64_t>(m) * n_out + n;
            for (int l = 0; l < lanes; ++l)
                v[l] = scalar_at(o, (base + l) % o.count);
            return _mm512_load_ps(v);
        }
    }
    __m512 v = load_f32(static_cast<const char *>(o.buf) + at * o.esz, o.dt, k);
    if (o.as_int && native_float_dt(o.dt))
        v = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(v));
    return v;
}

/// Saturating stores round to nearest even whatever the MXCSR mode.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline void store_f32(void *p, data_type_t dt, __m512 v, __mmask16 k) {
    constexpr int kRne = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    switch (dt) {
        case data_type_t::f32: _mm512_mask_storeu_ps(p, k, v); break;
        case data_type_t::bf16:
            _mm256_mask_storeu_epi16(p, k, avx512_f32_to_bf16_rne(v));
            break;
        case data_type_t::f16:
            _mm256_mask_storeu_epi16(p, k, _mm512_cvtps_ph(v, kRne));
            break;
        case data_type_t::u8:
            v = _mm512_min_ps(_mm512_max_ps(v, _mm512_setzero_ps()),
                    _mm512_set1_ps(255.0f));
            _mm512_mask_cvtusepi32_storeu_epi8(
                    p, k, _mm512_cvt_roundps_epi32(v, kRne));
            break;
        case data_type_t::s8:
            v = _mm512_min_ps(_mm512_max_ps(v, _mm512_set1_ps(-128.0f)),
                    _mm512_set1_ps(127.0f));
            _mm512_mask_cvtsepi32_storeu_epi8(
                    p, k, _mm512_cvt_roundps_epi32(v, kRne));
            break;
        case data_type_t::s32:
            // 2147483520 is the largest float below 2^31.
            v = _mm512_min_ps(_mm512_max_ps(v, _mm512_set1_ps(-2147483648.0f)),
                    _mm512_set1_ps(2147483520.0f));
            _mm512_mask_storeu_epi32(p, k, _mm512_cvt_roundps_epi32(v, kRne));
            break;
        default: break;
    }
}

/// The lanes of x that are NaN.
ZENDNNL_TARGET("avx512f")
inline __mmask16 nan_lanes(__m512 x) {
    return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
}

/// elu(x) = x for x > 0, alpha * (e^x - 1) otherwise.  avx512_exp turns a NaN
/// into a number, so NaN lanes take the scalar formula of the post-op
/// reference; below its clamp the vector form already gives the reference's
/// result.
ZENDNNL_TARGET("avx512f,fma")
inline __m512 elu_op(__m512 x, float alpha) {
    const __m512 y = _mm512_mask_mov_ps(
            _mm512_mul_ps(_mm512_set1_ps(alpha),
                    _mm512_sub_ps(avx512_exp(x), _mm512_set1_ps(1.0f))),
            _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OQ), x);
    const __mmask16 nans = nan_lanes(x);
    if (nans == 0) return y;
    return avx512_scalar_lanes(y, x, nans, [alpha](float v) {
        return v > 0 ? v : alpha * (std::exp(v) - 1.0f);
    });
}

/// avx512_sigmoid returns a number for a NaN and clamps e^-x below
/// -kAvx512ExpArgHi: those lanes take the scalar formula of the post-op
/// reference, which divides in double.
ZENDNNL_TARGET("avx512f,fma")
inline __m512 sigmoid_op(__m512 x) {
    const __m512 y = avx512_sigmoid(x);
    const __mmask16 tail = _mm512_cmp_ps_mask(
            x, _mm512_set1_ps(-kAvx512ExpArgHi), _CMP_NGE_UQ);
    if (tail == 0) return y;
    return avx512_scalar_lanes(y, x, tail, [](float v) {
        return static_cast<float>(1.0 / (1.0 + std::exp(-v)));
    });
}

/// Below this magnitude tanh(x) rounds to x itself, where the vector form
/// loses its bits to cancellation.
constexpr float kTanhIsArgBelow = 2.44140625e-4f;

/// Lanes below kTanhIsArgBelow return x, which keeps tiny values and the
/// sign of zero.  The clamps of avx512_tanh turn a NaN into a number: NaN
/// lanes take the tanh of the post-op reference.  Past the clamps of the
/// exponential the vector form already gives the reference's result.
ZENDNNL_TARGET("avx512f,fma")
inline __m512 tanh_op(__m512 x) {
    const __mmask16 tiny = _mm512_cmp_ps_mask(
            _mm512_abs_ps(x), _mm512_set1_ps(kTanhIsArgBelow), _CMP_LT_OQ);
    const __m512 y = _mm512_mask_mov_ps(avx512_tanh(x), tiny, x);
    const __mmask16 nans = nan_lanes(x);
    if (nans == 0) return y;
    return avx512_scalar_lanes(
            y, x, nans, [](float v) { return std::tanh(v); });
}

/// avx512_exp returns a number for a NaN and clamps x to [kAvx512ExpArgLo,
/// kAvx512ExpArgHi]: those lanes take the expf of the post-op reference,
/// which overflows and underflows past the float range.
ZENDNNL_TARGET("avx512f,fma")
inline __m512 exp_op(__m512 x) {
    const __m512 y = avx512_exp(x);
    const __mmask16 tail = static_cast<__mmask16>(
            _mm512_cmp_ps_mask(x, _mm512_set1_ps(kAvx512ExpArgLo), _CMP_NGE_UQ)
            | _mm512_cmp_ps_mask(
                    x, _mm512_set1_ps(kAvx512ExpArgHi), _CMP_NLE_UQ));
    if (tail == 0) return y;
    return avx512_scalar_lanes(y, x, tail, [](float v) { return std::exp(v); });
}

/// 0.5 x (1 + tanh(sqrt(2 / pi) x (1 + 0.044715 x^2))), the gelu_tanh
/// post-op and the gate of gelu_tanh_and_mul.
ZENDNNL_TARGET("avx512f,fma")
inline __m512 gelu_tanh(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 inner = _mm512_mul_ps(
            _mm512_mul_ps(_mm512_set1_ps(kSqrt2OverPi), x),
            _mm512_fmadd_ps(
                    _mm512_mul_ps(_mm512_set1_ps(kGeluTanhCoeff), x), x, one));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x),
            _mm512_add_ps(one, avx512_tanh(inner)));
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline __m512 eltwise(const native_postop_t &op, __m512 x, int lanes) {
    const __m512 zero = _mm512_setzero_ps();
    switch (op.type) {
        case post_op_type_t::relu: return _mm512_max_ps(x, zero);
        case post_op_type_t::leaky_relu:
            return _mm512_mask_mov_ps(
                    _mm512_mul_ps(x, _mm512_set1_ps(op.alpha)),
                    _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ), x);
        case post_op_type_t::elu: return elu_op(x, op.alpha);
        case post_op_type_t::gelu_tanh: return gelu_tanh(x);
        case post_op_type_t::gelu_erf: return avx512_gelu_erf(x);
        case post_op_type_t::sigmoid: return sigmoid_op(x);
        case post_op_type_t::swish: return avx512_swish(x, op.alpha);
        case post_op_type_t::tanh: return tanh_op(x);
        case post_op_type_t::square: return _mm512_mul_ps(x, x);
        case post_op_type_t::abs: return _mm512_abs_ps(x);
        case post_op_type_t::sqrt: return _mm512_sqrt_ps(x);
        case post_op_type_t::exp: return exp_op(x);
        case post_op_type_t::log: {
            alignas(64) float v[16];
            _mm512_store_ps(v, x);
            for (int l = 0; l < lanes; ++l)
                v[l] = std::log(v[l]);
            return _mm512_load_ps(v);
        }
        case post_op_type_t::clip:
            return _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(op.alpha)),
                    _mm512_set1_ps(op.beta));
        case post_op_type_t::mish: return avx512_mish(x);
        default: return x;
    }
}

/// Post-ops [from, to) at columns oc .. oc + lanes of row m of the output
/// the binary operands index, cols columns wide.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline __m512 run_ops(const native_epilogue_t &ep, size_t from, size_t to,
        __m512 v, int m, int oc, int cols, __mmask16 k, int lanes) {
    for (size_t i = from; i < to; ++i) {
        const native_postop_t &op = ep.ops[i];
        if (op.type == post_op_type_t::binary_add)
            v = _mm512_add_ps(
                    v, load_operand(op.operand, m, oc, cols, k, lanes));
        else if (op.type == post_op_type_t::binary_mul)
            v = _mm512_mul_ps(
                    v, load_operand(op.operand, m, oc, cols, k, lanes));
        else
            v = eltwise(op, v, lanes);
    }
    return v;
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline void store_out(
        const native_epilogue_t &ep, __m512 v, int m, int oc, int lanes) {
    const __mmask16 k = lane_mask(lanes);
    if (ep.requant) {
        // The reference rounds the product before adding the zero point.
        v = avx512_mul_unfused(
                v, load_operand(ep.dst_scale, m, oc, ep.n_out, k, lanes));
        if (ep.has_dst_zp)
            v = _mm512_add_ps(
                    v, load_operand(ep.dst_zp, m, oc, ep.n_out, k, lanes));
    }
    store_f32(dst_at(ep, m, oc), ep.dst_dt, v, k);
}

/// Output columns oc .. oc + lanes of row m: post-ops, then the store, or
/// the rows buffer when a softmax follows.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline void emit(
        const native_epilogue_t &ep, __m512 v, int m, int oc, int lanes) {
    const bool staged = ep.softmax_at < ep.ops.size();
    const __mmask16 k = lane_mask(lanes);
    v = run_ops(ep, 0, staged ? ep.softmax_at : ep.ops.size(), v, m, oc,
            ep.n_out, k, lanes);
    if (staged)
        _mm512_mask_storeu_ps(
                ep.rows.get() + static_cast<size_t>(m) * ep.n_out + oc, k, v);
    else
        store_out(ep, v, m, oc, lanes);
}

/// Shifted by the row maximum; the sum accumulates in double.
void softmax_row(float *row, int n) {
    float mx = row[0];
    for (int j = 1; j < n; ++j)
        mx = std::max(mx, row[j]);
    double sum = 0.0;
    for (int j = 0; j < n; ++j)
        sum += std::exp(row[j] - mx);
    for (int j = 0; j < n; ++j)
        row[j] = static_cast<float>(std::exp(row[j] - mx) / sum);
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void finish_row(const native_epilogue_t &ep, int m) {
    float *row = ep.rows.get() + static_cast<size_t>(m) * ep.n_out;
    for (size_t i = ep.softmax_at; i < ep.ops.size(); ++i) {
        if (ep.ops[i].type == post_op_type_t::softmax) {
            softmax_row(row, ep.n_out);
            continue;
        }
        for (int j = 0; j < ep.n_out; j += 16) {
            const int lanes = std::min(16, ep.n_out - j);
            const __mmask16 k = lane_mask(lanes);
            const __m512 v
                    = run_ops(ep, i, i + 1, _mm512_maskz_loadu_ps(k, row + j),
                            m, j, ep.n_out, k, lanes);
            _mm512_mask_storeu_ps(row + j, k, v);
        }
    }
    for (int j = 0; j < ep.n_out; j += 16) {
        const int lanes = std::min(16, ep.n_out - j);
        store_out(ep, _mm512_maskz_loadu_ps(lane_mask(lanes), row + j), m, j,
                lanes);
    }
}

/// The bias of weight columns p .. p + lanes as f32; the conversions are
/// exact.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline __m512 load_bias(const native_epilogue_t &ep, int p, __mmask16 k) {
    return load_f32(static_cast<const char *>(ep.bias_raw)
                    + static_cast<size_t>(p) * ep.bias_esz,
            ep.bias_dt, k);
}

/// The pre-activation columns p .. p + lanes of row m of beta_src, as f32;
/// the conversions are exact.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline __m512 load_pre_activation(
        const native_epilogue_t &ep, int m, int p, __mmask16 k) {
    const char *row = static_cast<const char *>(ep.beta_src)
            + static_cast<size_t>(m) * ep.beta_ld * ep.dst_esz;
    return load_f32(row + static_cast<size_t>(p) * ep.dst_esz, ep.dst_dt, k);
}

/// v = alpha * acc + beta * dst + bias at weight columns p .. p + lanes, in
/// the reference order, one rounding per operation (neither product is
/// fused with the add that follows it); a gated call reads beta's dst at
/// the pre-activation columns.
template <bool Gated>
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
inline __m512 pre(const native_epilogue_t &ep, const float *acc, int m, int p,
        int lanes) {
    const __mmask16 k = lane_mask(lanes);
    __m512 v = _mm512_maskz_loadu_ps(k, acc);
    if (ep.alpha != 1.0f) v = avx512_mul_unfused(v, _mm512_set1_ps(ep.alpha));
    if (ep.beta != 0.0f) {
        if constexpr (Gated)
            v = _mm512_add_ps(v,
                    avx512_mul_unfused(load_pre_activation(ep, m, p, k),
                            _mm512_set1_ps(ep.beta)));
        else
            v = _mm512_add_ps(v,
                    avx512_mul_unfused(load_f32(dst_at(ep, m, p), ep.dst_dt, k),
                            _mm512_set1_ps(ep.beta)));
    }
    if (ep.bias != nullptr)
        v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(k, ep.bias + p));
    else if (ep.bias_raw != nullptr)
        v = _mm512_add_ps(v, load_bias(ep, p, k));
    return v;
}

/// The pre-activation of weight columns p .. p + lanes of row m: pre, then
/// the post-ops, whose binary operands index the [M, N] pre-activation.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline __m512 pre_activation(const native_epilogue_t &ep, const float *acc,
        int m, int p, int lanes) {
    const __m512 v = pre<true>(ep, acc, m, p, lanes);
    if (ep.ops.empty()) return v;
    return run_ops(
            ep, 0, ep.ops.size(), v, m, p, ep.N, lane_mask(lanes), lanes);
}

/// The rows of a store_only epilogue: whole vectors stored unmasked, the
/// values the general path stores.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,fma")
void store_tile(const native_epilogue_t &ep, const float *C, int ldc_acc,
        int m0, int mr, int n0, int nr) {
    const bool bf16 = ep.dst_dt == data_type_t::bf16;
    const int full = nr & ~15;
    for (int i = 0; i < mr; ++i) {
        const float *acc = C + static_cast<size_t>(i) * ldc_acc;
        char *row = static_cast<char *>(dst_at(ep, m0 + i, n0));
        for (int j = 0; j < full; j += 16) {
            const __m512 v = _mm512_loadu_ps(acc + j);
            if (bf16)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(
                                            row + static_cast<size_t>(j) * 2),
                        avx512_f32_to_bf16_rne(v));
            else
                _mm512_storeu_ps(row + static_cast<size_t>(j) * 4, v);
        }
        if (full < nr) {
            const __mmask16 k = lane_mask(nr - full);
            store_f32(row + static_cast<size_t>(full) * ep.dst_esz, ep.dst_dt,
                    _mm512_maskz_loadu_ps(k, acc + full), k);
        }
    }
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
inline __m512 gated_value(matmul_gated_act_t gated, __m512 g, __m512 u) {
    switch (gated) {
        case matmul_gated_act_t::silu_and_mul:
            return _mm512_mul_ps(avx512_silu(g), u);
        case matmul_gated_act_t::gelu_and_mul:
            return _mm512_mul_ps(avx512_gelu_erf(g), u);
        case matmul_gated_act_t::gelu_tanh_and_mul:
            return _mm512_mul_ps(gelu_tanh(g), u);
        default: return avx512_swiglu_oai(g, u);
    }
}

/// The outputs j0 .. j1 of a staged row m of a gated call over [gate | up]
/// halves: the activation of its staged pre-activation columns i and N / 2
/// + i.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void finish_gated_row(const native_epilogue_t &ep, int m, int j0, int j1) {
    const float *row = ep.staged + static_cast<size_t>(m) * ep.N;
    const int half = ep.N / 2;
    for (int j = j0; j < j1; j += 16) {
        const int lanes = std::min(16, j1 - j);
        const __mmask16 k = lane_mask(lanes);
        store_out(ep,
                gated_value(ep.gated, _mm512_maskz_loadu_ps(k, row + j),
                        _mm512_maskz_loadu_ps(k, row + half + j)),
                m, j, lanes);
    }
}

struct thread_staged_t {
    std::unique_ptr<float[]> buf;
    size_t cap = 0;
};

thread_staged_t &thread_staged() {
    static thread_local thread_staged_t staged;
    return staged;
}

/// The calling thread's staging rows of at least floats floats, kept for
/// the thread's lifetime: grown to at least twice their size up to
/// kNativeThreadStagedFloats, or to floats above it; rows above that bound
/// are freed by the thread's next request that fits it.  nullptr when they
/// cannot be allocated.
float *thread_staged_rows(size_t floats) {
    thread_staged_t &t = thread_staged();
    const size_t bound = kNativeThreadStagedFloats;
    const bool shrink = t.cap > bound && floats <= bound;
    if (t.cap < floats || shrink) {
        const size_t cap = floats > bound
                ? floats
                : std::min(bound, std::max(floats, shrink ? 0 : 2 * t.cap));
        t.buf.reset();
        t.buf.reset(new (std::nothrow) float[cap]);
        t.cap = t.buf ? cap : 0;
    }
    return t.buf.get();
}

} // namespace

size_t native_thread_staged_floats() {
    return thread_staged().cap;
}

status_t prepare_epilogue(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, void *dst, bool paired,
        native_epilogue_t &ep, const char **why, bool buffers) {
    const matmul_gated_act_t gated = desc.gated_act;
    const bool is_gated = gated != matmul_gated_act_t::none;
    const bool halves = native_gated_halves(gated);
    if (is_gated) {
        if (!halves && gated != matmul_gated_act_t::swiglu_oai_mul)
            return turned_away(status_t::unimplemented,
                    "an unknown gated activation", why);
        if (desc.N % 2 != 0)
            return turned_away(status_t::failure,
                    "a gated activation over an odd number of columns", why);
        // swiglu weights interleave their pairs: their tiles are plain.
        if (paired && !halves)
            return turned_away(status_t::failure,
                    "paired tiles for a gated activation of interleaved pairs",
                    why);
        // beta reads the dst's N pre-activation columns.
        if (desc.beta != 0.0f && desc.ldc < desc.N)
            return turned_away(status_t::unimplemented,
                    "beta with a gated activation over a dst of fewer than N "
                    "columns",
                    why);
    } else if (paired) {
        return turned_away(status_t::unimplemented,
                "paired tiles without a gated activation", why);
    }
    switch (desc.dst_dt) {
        case data_type_t::f32:
        case data_type_t::bf16:
        case data_type_t::f16:
        case data_type_t::u8:
        case data_type_t::s8:
        case data_type_t::s32: break;
        default:
            return turned_away(status_t::unimplemented,
                    "a dst other than f32, bf16, f16, u8, s8 or s32", why);
    }

    ep.N = desc.N;
    ep.n_out = is_gated ? desc.N / 2 : desc.N;
    if (desc.ldc < ep.n_out || desc.M < 0)
        return turned_away(
                status_t::failure, "ldc below the output columns", why);
    ep.alpha = desc.alpha;
    ep.beta = desc.beta;
    ep.gated = gated;
    ep.paired = paired;
    ep.staged_halves = halves && !paired;
    ep.dst = dst;
    ep.dst_dt = desc.dst_dt;
    ep.dst_esz = static_cast<int>(size_of(desc.dst_dt));
    ep.ldc = desc.ldc;
    ep.beta_src = dst;
    ep.beta_ld = desc.ldc;
    // An s4 weight is always per group: its alpha and beta apply here.
    const bool is_int8
            = desc.wei_dt == data_type_t::s8 || desc.wei_dt == data_type_t::s4;
    if (desc.wei_dt == data_type_t::s8
            && !has_per_group_wei_scale(params.quant_params, desc.N)
            && (desc.alpha != 1.0f || desc.beta != 0.0f))
        return turned_away(status_t::unimplemented,
                "INT8 alpha / beta outside the INT8 tile", why);

    ep.bias = nullptr;
    ep.bias_raw = nullptr;
    if (bias != nullptr) {
        const data_type_t bdt = params.dtypes.bias;
        if (!is_operand_dt(bdt))
            return turned_away(status_t::unimplemented,
                    "a bias other than f32, bf16, f16, s8, u8 or s32", why);
        if (bdt == data_type_t::f32) {
            ep.bias = static_cast<const float *>(bias);
        } else {
            ep.bias_raw = bias;
            ep.bias_dt = bdt;
            ep.bias_esz = static_cast<int>(size_of(bdt));
        }
    }

    ep.ops.clear();
    ep.ops.reserve(params.postop_.size());
    ep.softmax_at = 0;
    bool softmax = false;
    // The binary operands of a gated call's post-ops index its
    // pre-activation.
    const int op_cols = is_gated ? desc.N : ep.n_out;
    for (const matmul_post_op &po : params.postop_) {
        native_postop_t op;
        op.type = po.po_type;
        switch (po.po_type) {
            case post_op_type_t::none: continue;
            case post_op_type_t::binary_add:
            case post_op_type_t::binary_mul: {
                const int64_t count = checked_quant_param_num_elements(po.dims);
                const status_t st = make_operand(po.buff, po.dtype, count,
                        po.leading_dim, desc.M, op_cols, false, op.operand);
                if (st != status_t::success)
                    return turned_away(st,
                            "a binary post-op operand of a type or shape the "
                            "epilogue does not read",
                            why);
                break;
            }
            case post_op_type_t::softmax:
                if (is_gated)
                    return turned_away(status_t::unimplemented,
                            "a softmax with a gated activation", why);
                if (!softmax) ep.softmax_at = ep.ops.size();
                softmax = true;
                break;
            case post_op_type_t::relu:
            case post_op_type_t::leaky_relu:
            case post_op_type_t::elu:
            case post_op_type_t::gelu_tanh:
            case post_op_type_t::gelu_erf:
            case post_op_type_t::sigmoid:
            case post_op_type_t::swish:
            case post_op_type_t::tanh:
            case post_op_type_t::square:
            case post_op_type_t::abs:
            case post_op_type_t::sqrt:
            case post_op_type_t::exp:
            case post_op_type_t::log:
            case post_op_type_t::clip:
            case post_op_type_t::mish:
                op.alpha = po.alpha_or_default();
                op.beta = po.beta;
                break;
            default:
                return turned_away(status_t::unimplemented,
                        "a post-op the epilogue does not apply", why);
        }
        ep.ops.push_back(op);
    }
    if (!softmax) ep.softmax_at = ep.ops.size();

    ep.rows.reset();
    ep.row_cols.reset();
    ep.staged = nullptr;
    if (softmax && desc.M > 0 && buffers) {
        const size_t cells = static_cast<size_t>(desc.M) * ep.n_out;
        ep.rows.reset(new (std::nothrow) float[cells]);
        ep.row_cols.reset(new (std::nothrow) std::atomic<int>[desc.M]);
        if (!ep.rows || !ep.row_cols) {
            log_error("native epilogue: no rows buffer of ", cells, " floats");
            return status_t::failure;
        }
        for (int m = 0; m < desc.M; ++m)
            ep.row_cols[m].store(0, std::memory_order_relaxed);
    }
    if (ep.staged_halves && desc.M > 0 && buffers) {
        const size_t cells = static_cast<size_t>(desc.M) * desc.N;
        ep.staged = thread_staged_rows(cells);
        if (ep.staged == nullptr) {
            log_error("native epilogue: no rows buffer of ", cells, " floats");
            return status_t::failure;
        }
    }

    const auto &qp = params.quant_params;
    ep.requant = is_int8 && qp.dst_scale.buff != nullptr;
    ep.has_dst_zp = false;
    if (ep.requant) {
        status_t st = make_operand(qp.dst_scale.buff, qp.dst_scale.dt,
                checked_quant_param_num_elements(qp.dst_scale.dims), 0, desc.M,
                ep.n_out, false, ep.dst_scale);
        const int64_t zp_count
                = checked_quant_param_num_elements(qp.dst_zp.dims);
        ep.has_dst_zp = qp.dst_zp.buff != nullptr && zp_count != 0;
        if (st == status_t::success && ep.has_dst_zp)
            st = make_operand(qp.dst_zp.buff, qp.dst_zp.dt, zp_count, 0, desc.M,
                    ep.n_out, true, ep.dst_zp);
        if (st == status_t::unimplemented)
            return turned_away(st,
                    "a dst scale or zero point of a type the epilogue does not "
                    "read",
                    why);
        if (st != status_t::success)
            return turned_away(st, "a bad dst scale or zero point", why);
    }
    ep.store_only = !is_gated && ep.alpha == 1.0f && ep.beta == 0.0f
            && ep.bias == nullptr && ep.bias_raw == nullptr && ep.ops.empty()
            && !ep.requant
            && (ep.dst_dt == data_type_t::f32
                    || ep.dst_dt == data_type_t::bf16);
    return status_t::success;
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512dq,fma")
void apply_epilogue_tile(const native_epilogue_t &ep, float *C, int ldc_acc,
        int m0, int mr, int n0, int nr) {
    if (mr <= 0 || nr <= 0) return;
    if (ep.store_only) {
        store_tile(ep, C, ldc_acc, m0, mr, n0, nr);
        return;
    }
    const bool gated = ep.gated != matmul_gated_act_t::none;
    if (ep.paired) {
        // Column j of the tile's first half is gate column n0 / 2 + j, and
        // column half + j the up column paired with it.
        const int half = nr / 2;
        const int i0 = n0 / 2;
        const int up0 = ep.N / 2 + i0;
        for (int i = 0; i < mr; ++i) {
            const int m = m0 + i;
            const float *acc = C + static_cast<size_t>(i) * ldc_acc;
            for (int j = 0; j < half; j += 16) {
                const int lanes = std::min(16, half - j);
                const __m512 g = pre_activation(ep, acc + j, m, i0 + j, lanes);
                const __m512 u
                        = pre_activation(ep, acc + half + j, m, up0 + j, lanes);
                store_out(ep, gated_value(ep.gated, g, u), m, i0 + j, lanes);
            }
        }
        return;
    }
    if (ep.staged_halves) {
        // The pre-activation of whole rows, paired by pair_staged_halves.
        for (int i = 0; i < mr; ++i) {
            const int m = m0 + i;
            const float *acc = C + static_cast<size_t>(i) * ldc_acc;
            float *row = ep.staged + static_cast<size_t>(m) * ep.N;
            for (int j = 0; j < nr; j += 16) {
                const int lanes = std::min(16, nr - j);
                _mm512_mask_storeu_ps(row + n0 + j, lane_mask(lanes),
                        pre_activation(ep, acc + j, m, n0 + j, lanes));
            }
        }
        return;
    }
    const int width = gated ? nr / 2 : nr;
    const __m512i gate_lanes = _mm512_load_si512(kGateLanes);
    const __m512i up_lanes = _mm512_load_si512(kUpLanes);
    for (int i = 0; i < mr; ++i) {
        const int m = m0 + i;
        const float *acc = C + static_cast<size_t>(i) * ldc_acc;
        if (!gated) {
            for (int j = 0; j < nr; j += 16) {
                const int lanes = std::min(16, nr - j);
                emit(ep, pre<false>(ep, acc + j, m, n0 + j, lanes), m, n0 + j,
                        lanes);
            }
        } else {
            // swiglu: the post-ops act on the pre-activation at weight
            // columns 2i and 2i + 1, before the pairs separate.
            for (int j = 0; j < nr; j += 32) {
                const int lanes = std::min(32, nr - j);
                const int hi_lanes = lanes - 16;
                const int i0 = (n0 + j) / 2;
                const int pairs = lanes / 2;
                const __m512 lo = pre_activation(
                        ep, acc + j, m, n0 + j, std::min(16, lanes));
                const __m512 hi = hi_lanes > 0
                        ? pre_activation(
                                  ep, acc + j + 16, m, n0 + j + 16, hi_lanes)
                        : _mm512_setzero_ps();
                const __m512 g = _mm512_permutex2var_ps(lo, gate_lanes, hi);
                const __m512 u = _mm512_permutex2var_ps(lo, up_lanes, hi);
                store_out(ep, gated_value(ep.gated, g, u), m, i0, pairs);
            }
        }
        if (ep.softmax_at < ep.ops.size()
                && ep.row_cols[m].fetch_add(width, std::memory_order_acq_rel)
                                + width
                        == ep.n_out)
            finish_row(ep, m);
    }
}

namespace {

/// Units of the staged pairing: a row and up to 256 of its outputs.
constexpr int kStagedPairUnit = 256;

} // namespace

int64_t staged_pair_units(const native_epilogue_t &ep, int M) {
    const int per_row = (ep.N / 2 + kStagedPairUnit - 1) / kStagedPairUnit;
    return static_cast<int64_t>(M) * per_row;
}

void pair_staged_halves(const native_epilogue_t &ep, int M, int i, int n) {
    const int half = ep.N / 2;
    const int unit = kStagedPairUnit;
    const int per_row = (half + unit - 1) / unit;
    const int64_t units = static_cast<int64_t>(M) * per_row;
    for (int64_t u = units * i / n; u < units * (i + 1) / n; ++u) {
        const int m = static_cast<int>(u / per_row);
        const int j0 = static_cast<int>(u % per_row) * unit;
        finish_gated_row(ep, m, j0, std::min(half, j0 + unit));
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
