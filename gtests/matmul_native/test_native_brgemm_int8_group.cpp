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

// INT8 per-group weight scales ({G, N} along K) on the native BRGEMM
// (ALGO 10): matmul_direct calls with a per-group {M, G}, per-token or
// per-tensor source scale over a static s8 source or a bf16 / f32 source
// quantized in the call, into an f32 / bf16 dst or a u8 / s8 / s32 dst (with
// or without a dst scale and zero point).  Every case checks that the native
// BRGEMM served the call, then compares it with
//   * the reference algorithm's order modelled here (the exact integer sum
//     of each group, then f32(sum) * src scale * wei scale added in group
//     order, then the dst scale and zero point and the rounding of an
//     integer dst), bit for bit;
//   * the reference algorithm itself for a per-group source, bit for bit;
//   * an f64 model within its rounding bound for a per-token or per-tensor
//     source (the reference algorithm sums such a call over all of K);
//   * the default route within the sym-quant band, when the process is not
//     pinned to an algorithm.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/backends/reference/reference_kernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile_internal.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"
#include "memory/memory_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::read_and_cast;
using zendnnl::memory::status_t;
using quant_t = mm::matmul_quantization_params_t::matmul_quant_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kNone = data_type_t::none;
constexpr float kSqrt2Over2 = 0.707106769084930419921875f;

/// Static s8, bf16 / f32 quantized in the call (to s8, or to u8 with zp),
/// or static u8.
enum class src_t { s8, bf16, f32, u8 };
/// Source scale: {1, 1}, {M, 1} or {M, G}.
enum class gran_t { tensor, token, group };
/// Weight scale: {G, N}, or with an {M, G} source scale {1, N}, {1, 1} or
/// none, which apply after the groups.
enum class wgran_t { group, channel, tensor, none };
/// Post-ops: binary add of an [M, N] f32 matrix, gelu (erf), or both.
enum class po_t { none, add, gelu, gelu_add };

struct gcase_t {
    std::string name;
    int M = 16, N = 64, gs = 32, G = 4; ///< K = gs * G
    src_t src = src_t::s8;
    gran_t gran = gran_t::group;
    wgran_t wgran = wgran_t::group;
    bool transA = false, transB = false;
    data_type_t dst = kF32, bias = kNone;
    data_type_t wei_scale_dt = kF32, src_scale_dt = kF32;
    po_t po = po_t::none;
    /// A per-tensor dst scale and zero point.
    bool requant = false;
    /// A per-tensor source zero point: of a static u8 source (drawn from
    /// zp_lo .. zp_lo + 70), or of a bf16 / f32 source quantized to u8.
    bool zp = false;
    int32_t zp_lo = 90;
    /// The values of a bf16 / f32 source.
    float src_lo = -1.0f, src_hi = 1.0f;
    float alpha = 1.0f, beta = 0.0f;
    int wc = 1; ///< weight cache mode of the call
    int threads = 4; ///< 0: every thread
    int calls = 1; ///< native calls on the same weight
    /// The case tests the tiles' batches of groups: it is skipped where the
    /// plan of the call takes all groups in one batch.
    bool batches = false;
    int K() const { return gs * G; }
};

void PrintTo(const gcase_t &c, std::ostream *os) {
    *os << c.name;
}

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kF32: return "f32";
        case kBf16: return "bf16";
        case kF16: return "f16";
        case kS8: return "s8";
        case kU8: return "u8";
        case kS32: return "s32";
        default: return "none";
    }
}

bool integer_dt(data_type_t dt) {
    return dt == kS8 || dt == kU8 || dt == kS32;
}

std::string describe(const gcase_t &c) {
    static const char *const kSrc[] = {"s8", "dynbf16", "dynf32", "u8"};
    static const char *const kGran[] = {"tensor", "token", "group"};
    static const char *const kPo[] = {"", "_add", "_gelu", "_geluadd"};
    std::string s = std::string(kSrc[static_cast<int>(c.src)]) + "_"
            + kGran[static_cast<int>(c.gran)] + "_m" + std::to_string(c.M)
            + "_n" + std::to_string(c.N) + "_gs" + std::to_string(c.gs) + "x"
            + std::to_string(c.G) + "_" + dt_name(c.dst);
    if (c.zp) s += "_zp" + std::to_string(c.zp_lo);
    if (c.src_lo != -1.0f || c.src_hi != 1.0f)
        s += "_from" + std::to_string(static_cast<int>(c.src_lo));
    static const char *const kWgran[] = {"", "_wchan", "_wtensor", "_wnone"};
    s += kWgran[static_cast<int>(c.wgran)];
    if (c.transA) s += "_ta";
    if (c.transB) s += "_tb";
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    s += kPo[static_cast<int>(c.po)];
    if (c.requant) s += "_rq";
    if (c.alpha != 1.0f || c.beta != 0.0f)
        s += "_a" + std::to_string(static_cast<int>(c.alpha * 100)) + "b"
                + std::to_string(static_cast<int>(c.beta * 100));
    if (c.wei_scale_dt != kF32)
        s += std::string("_ws") + dt_name(c.wei_scale_dt);
    if (c.src_scale_dt != kF32)
        s += std::string("_ss") + dt_name(c.src_scale_dt);
    s += "_wc" + std::to_string(c.wc) + "_t" + std::to_string(c.threads);
    if (c.calls != 1) s += "_x" + std::to_string(c.calls);
    for (char &ch : s)
        if (ch == '-') ch = 'm';
    return s;
}

uint32_t seed_of(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char ch : s)
        h = (h ^ static_cast<uint8_t>(ch)) * 16777619u;
    return h;
}

void set_quant(quant_t &q, const void *buff, data_type_t dt,
        std::vector<int64_t> dims) {
    q.buff = buff;
    q.dt = dt;
    q.dims = std::move(dims);
}

buffer_t copy_of(const buffer_t &b) {
    buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

int64_t src_scale_count(const gcase_t &c) {
    return c.gran == gran_t::tensor   ? 1
            : c.gran == gran_t::token ? c.M
                                      : static_cast<int64_t>(c.M) * c.G;
}

struct operands_t {
    int lda = 0, ldb = 0;
    buffer_t src, wei, bias, wei_scale, src_scale, src_zp, bin, init, dst_scale,
            dst_zp;
    mm::matmul_params params;
};

operands_t make_operands(const gcase_t &c) {
    operands_t o;
    const uint32_t seed = seed_of(c.name);
    const int M = c.M, N = c.N, K = c.K();
    o.lda = c.transA ? M : K;
    o.ldb = c.transB ? K : N;
    const size_t a_n = static_cast<size_t>(M) * K;
    const data_type_t src_dt = c.src == src_t::s8 ? kS8
            : c.src == src_t::u8                  ? kU8
            : c.src == src_t::bf16                ? kBf16
                                                  : kF32;
    o.src = src_dt == kS8 ? nt::random_buffer(kS8, a_n, seed, -128, 127)
            : src_dt == kU8
            ? nt::random_buffer(kU8, a_n, seed, 0, 255)
            : nt::random_buffer(src_dt, a_n, seed, c.src_lo, c.src_hi);
    o.wei = nt::random_buffer(
            kS8, static_cast<size_t>(K) * N, seed + 1, -128, 127);
    const int64_t n_wei_scale = c.wgran == wgran_t::group ? int64_t {c.G} * N
            : c.wgran == wgran_t::channel                 ? N
                                                          : 1;
    o.wei_scale = nt::random_buffer(
            c.wei_scale_dt, n_wei_scale, seed + 2, 0.002f, 0.01f);
    mm::matmul_params &p = o.params;
    p.dtypes.src = src_dt;
    p.dtypes.wei = kS8;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    auto &qp = p.quant_params;
    if (c.wgran != wgran_t::none)
        set_quant(qp.wei_scale, o.wei_scale.get(), c.wei_scale_dt,
                c.wgran == wgran_t::group ? std::vector<int64_t> {c.G, N}
                        : c.wgran == wgran_t::channel
                        ? std::vector<int64_t> {1, N}
                        : std::vector<int64_t> {1, 1});
    const int64_t n_scale = src_scale_count(c);
    const bool static_src = c.src == src_t::s8 || c.src == src_t::u8;
    if (static_src) {
        o.src_scale = nt::random_buffer(
                c.src_scale_dt, n_scale, seed + 3, 0.002f, 0.01f);
    } else {
        p.dynamic_quant = true;
        p.dtypes.compute = c.zp ? kU8 : kS8;
        o.src_scale = nt::make_buffer(n_scale * nt::dt_bytes(c.src_scale_dt));
    }
    if (c.zp) {
        o.src_zp = static_src ? nt::random_buffer(kS32, 1, seed + 9,
                                        static_cast<float>(c.zp_lo),
                                        static_cast<float>(c.zp_lo + 70))
                              : nt::make_buffer(sizeof(int32_t));
        set_quant(qp.src_zp, o.src_zp.get(), kS32, {1, 1});
    }
    set_quant(qp.src_scale, o.src_scale.get(), c.src_scale_dt,
            c.gran == gran_t::tensor          ? std::vector<int64_t> {1, 1}
                    : c.gran == gran_t::token ? std::vector<int64_t> {M, 1}
                                              : std::vector<int64_t> {M, c.G});
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, N, seed + 4, -1.0f, 1.0f);
    if (c.po == po_t::gelu || c.po == po_t::gelu_add) {
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::gelu_erf;
        p.postop_.push_back(po);
    }
    if (c.po == po_t::add || c.po == po_t::gelu_add) {
        o.bin = nt::random_buffer(
                kF32, static_cast<size_t>(M) * N, seed + 5, -1.0f, 1.0f);
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::binary_add;
        po.buff = o.bin.get();
        po.dtype = kF32;
        po.dims = {M, N};
        p.postop_.push_back(po);
    }
    if (c.requant) {
        o.dst_scale = nt::random_buffer(kF32, 1, seed + 7, 0.5f, 2.0f);
        o.dst_zp = nt::random_buffer(kS32, 1, seed + 8,
                c.dst == kU8 ? 100.0f : -10.0f, c.dst == kU8 ? 130.0f : 10.0f);
        set_quant(qp.dst_scale, o.dst_scale.get(), kF32, {1, 1});
        set_quant(qp.dst_zp, o.dst_zp.get(), kS32, {1, 1});
    }
    const float init_lo = !integer_dt(c.dst) ? -2.0f
            : c.dst == kU8                   ? 0.0f
                                             : -20.0f;
    o.init = nt::random_buffer(c.dst, static_cast<size_t>(M) * N, seed + 6,
            init_lo, init_lo + (integer_dt(c.dst) ? 40.0f : 4.0f));
    return o;
}

/// The call on weight wei into dst; a source scale buffer per call, since a
/// dynamic quantization writes its scales there.
nt::call_t make_call(const gcase_t &c, const operands_t &o, const void *wei,
        void *dst, const buffer_t &src_scale) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K();
    call.transA = c.transA;
    call.transB = c.transB;
    call.lda = o.lda;
    call.ldb = o.ldb;
    call.ldc = c.N;
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = wei;
    call.bias = c.bias == kNone ? nullptr : o.bias.get();
    call.dst = dst;
    call.params = o.params;
    call.params.quant_params.src_scale.buff = src_scale.get();
    return call;
}

/// The s8 / u8 source rows [M][K], their zero point and their scales [M][G]
/// the native route multiplies: a static source as given, a dynamic one
/// quantized here by the shared quantizer from its rows made row-major, as
/// the route does.
struct source_t {
    std::vector<int32_t> a;
    int32_t zp = 0;
    std::vector<float> scale;
};

source_t source_of(const gcase_t &c, const operands_t &o) {
    const int M = c.M, K = c.K(), G = c.G;
    source_t s;
    s.a.resize(static_cast<size_t>(M) * K);
    s.scale.resize(static_cast<size_t>(M) * G);
    const auto at = [&](int m, int k) {
        return c.transA ? static_cast<size_t>(k) * o.lda + m
                        : static_cast<size_t>(m) * o.lda + k;
    };
    const void *scales = o.src_scale.get();
    buffer_t written, written_zp;
    if (c.src == src_t::s8 || c.src == src_t::u8) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                s.a[static_cast<size_t>(m) * K + k] = c.src == src_t::s8
                        ? o.src.as<int8_t>()[at(m, k)]
                        : o.src.as<uint8_t>()[at(m, k)];
        if (c.zp) s.zp = o.src_zp.as<int32_t>()[0];
    } else {
        const data_type_t dt = o.params.dtypes.src;
        const size_t es = nt::dt_bytes(dt);
        buffer_t rows = nt::make_buffer(static_cast<size_t>(M) * K * es);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                std::memcpy(rows.as<uint8_t>()
                                + (static_cast<size_t>(m) * K + k) * es,
                        o.src.as<uint8_t>() + at(m, k) * es, es);
        written = copy_of(o.src_scale);
        mm::matmul_params q = o.params;
        q.quant_params.src_scale.buff = written.get();
        if (c.zp) {
            written_zp = copy_of(o.src_zp);
            q.quant_params.src_zp.buff = written_zp.get();
        }
        const void *src = rows.get();
        int quantized_lda = K;
        size_t elem_size = es;
        mm::matmul_batch_params_t batch;
        mm::reorder_quant_buffers_t buffers;
        EXPECT_EQ(mm::reorder_quantization_wrapper(src, K, quantized_lda,
                          elem_size, q, batch, false, M, K, 1, buffers),
                status_t::success);
        EXPECT_EQ(q.dtypes.src, c.zp ? kU8 : kS8);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(m) * quantized_lda + k;
                s.a[static_cast<size_t>(m) * K + k] = c.zp
                        ? static_cast<const uint8_t *>(src)[i]
                        : static_cast<const int8_t *>(src)[i];
            }
        if (c.zp) s.zp = written_zp.as<int32_t>()[0];
        scales = written.get();
    }
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < G; ++g) {
            const size_t i = c.gran == gran_t::tensor ? 0
                    : c.gran == gran_t::token         ? static_cast<size_t>(m)
                                              : static_cast<size_t>(m) * G + g;
            s.scale[static_cast<size_t>(m) * G + g]
                    = nt::load_f32(scales, c.src_scale_dt, i);
        }
    return s;
}

int8_t weight_at(const gcase_t &c, const operands_t &o, int k, int n) {
    return o.wei.as<int8_t>()[c.transB ? static_cast<size_t>(n) * o.ldb + k
                                       : static_cast<size_t>(k) * o.ldb + n];
}

/// Element (m, n) of the call before its store, in the reference
/// algorithm's order: one rounding per operation.
float model_at(const gcase_t &c, const operands_t &o, const source_t &s, int m,
        int n) {
    const int K = c.K(), G = c.G, gs = c.gs;
    float sum = 0.0f;
    for (int g = 0; g < G; ++g) {
        int64_t d = 0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
            d += static_cast<int64_t>(
                         s.a[static_cast<size_t>(m) * K + k] - s.zp)
                    * weight_at(c, o, k, n);
        float v = static_cast<float>(d);
        v = v * s.scale[static_cast<size_t>(m) * G + g];
        if (c.wgran == wgran_t::group)
            v = v
                    * nt::load_f32(o.wei_scale.get(), c.wei_scale_dt,
                            static_cast<size_t>(g) * c.N + n);
        sum = sum + v;
    }
    const size_t mn = static_cast<size_t>(m) * c.N + n;
    // beta * C into an f32 or bf16 dst follows the weight scales; into any
    // other it comes before them.
    const bool beta_last = c.dst == kF32 || c.dst == kBf16;
    if (c.alpha != 1.0f) sum *= c.alpha;
    if (c.beta != 0.0f && !beta_last)
        sum += nt::load_f32(o.init.get(), c.dst, mn) * c.beta;
    if (c.wgran == wgran_t::channel || c.wgran == wgran_t::tensor)
        sum *= nt::load_f32(o.wei_scale.get(), c.wei_scale_dt,
                c.wgran == wgran_t::channel ? static_cast<size_t>(n) : 0);
    if (c.beta != 0.0f && beta_last)
        sum += nt::load_f32(o.init.get(), c.dst, mn) * c.beta;
    if (c.bias != kNone) sum += nt::load_f32(o.bias.get(), c.bias, n);
    if (c.po == po_t::gelu || c.po == po_t::gelu_add)
        sum = 0.5f * sum * (1.0f + std::erf(sum * kSqrt2Over2));
    if (c.po == po_t::add || c.po == po_t::gelu_add)
        sum = sum + o.bin.as<float>()[mn];
    return sum;
}

/// The same call in f64, and the bound of its f32 rounding: per group two
/// products and one sum, then alpha, beta and the bias, of relative error
/// 2^-24 each over the magnitudes they add.
void f64_at(const gcase_t &c, const operands_t &o, const source_t &s, int m,
        int n, double &value, double &bound) {
    const int K = c.K(), G = c.G, gs = c.gs;
    double sum = 0.0, mag = 0.0;
    for (int g = 0; g < G; ++g) {
        int64_t d = 0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
            d += static_cast<int64_t>(
                         s.a[static_cast<size_t>(m) * K + k] - s.zp)
                    * weight_at(c, o, k, n);
        const double v = static_cast<double>(d)
                * s.scale[static_cast<size_t>(m) * G + g]
                * nt::load_f32(o.wei_scale.get(), c.wei_scale_dt,
                        static_cast<size_t>(g) * c.N + n);
        sum += v;
        mag += std::fabs(v);
    }
    const size_t mn = static_cast<size_t>(m) * c.N + n;
    const double cin = c.beta != 0.0f
            ? static_cast<double>(nt::load_f32(o.init.get(), c.dst, mn))
                    * c.beta
            : 0.0;
    const double b
            = c.bias != kNone ? nt::load_f32(o.bias.get(), c.bias, n) : 0.0;
    value = sum * c.alpha + cin + b;
    bound = std::ldexp(1.0, -24) * (G + 4)
            * (mag * std::fabs(c.alpha) + std::fabs(cin) + std::fabs(b)) * 2.0;
}

/// Element i of a dst of type dt as the reference algorithm stores v: the
/// dst scale and zero point of an INT8 call (scale, then zero point, one
/// rounding each), then an integer dst clamped to its range and rounded to
/// nearest even.
void store_ref(
        const operands_t &o, void *out, data_type_t dt, size_t i, float v) {
    const auto &qp = o.params.quant_params;
    if (qp.dst_scale.buff != nullptr) {
        v = v * nt::load_f32(qp.dst_scale.buff, qp.dst_scale.dt, 0);
        if (qp.dst_zp.buff != nullptr)
            v = v
                    + static_cast<float>(read_and_cast<int32_t>(
                            qp.dst_zp.buff, qp.dst_zp.dt, 0));
    }
    if (!integer_dt(dt)) {
        nt::store_f32(out, dt, i, v);
        return;
    }
    const float lo = dt == kU8 ? 0.0f : dt == kS8 ? -128.0f : -2147483648.0f;
    const float hi = dt == kU8 ? 255.0f : dt == kS8 ? 127.0f : 2147483520.0f;
    nt::store_f32(out, dt, i, std::nearbyint(std::min(std::max(v, lo), hi)));
}

/// The values the call stores, in dst order.
buffer_t model_dst(const gcase_t &c, const operands_t &o, const source_t &s) {
    buffer_t out = nt::make_buffer(
            static_cast<size_t>(c.M) * c.N * nt::dt_bytes(c.dst));
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n)
            store_ref(o, out.get(), c.dst, static_cast<size_t>(m) * c.N + n,
                    model_at(c, o, s, m, n));
    return out;
}

bool transcendental(const gcase_t &c) {
    return c.po == po_t::gelu || c.po == po_t::gelu_add;
}

/// got against want where the post-op is the epilogue's gelu, whose error
/// against erf is the band of the epilogue suite; bf16 adds half an ulp.
::testing::AssertionResult near_gelu(
        const gcase_t &c, const void *got, const void *want) {
    return nt::within_band(got, c.N, want, c.N, c.dst, c.M, c.N,
            c.dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f, 3e-4f);
}

nat::native_problem_t problem_of(const nt::call_t &c) {
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda_or_tight();
    desc.ldb = c.ldb_or_tight();
    desc.ldc = c.ldc_or_tight();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = c.params.dtypes.src;
    desc.wei_dt = c.params.dtypes.wei;
    desc.dst_dt = c.params.dtypes.dst;
    desc.bias_dt = c.params.dtypes.bias;
    desc.bias = c.bias;
    desc.num_threads = c.params.num_threads;
    return nat::make_native_problem(desc, c.params, c.bias,
            nat::kNativeLayoutPackNr,
            nat::decide_int8_call(desc, c.params, c.bias));
}

/// Whether the per-group tiles of c, at the process planner's plan, run in
/// more than one batch of groups at each of their call widths, as
/// compute_group_tile cuts them (native_int8_group_span).
bool tiles_take_batches(const gcase_t &c, const operands_t &o) {
    const buffer_t scale = copy_of(o.src_scale);
    const nat::native_plan_t plan = nat::native_planner().plan(
            problem_of(make_call(c, o, o.wei.get(), nullptr, scale)));
    const int rows = std::min(plan.mb, c.M);
    for (int col = 0; col < plan.nr;) {
        const int cw = nat::native_int8_col_width(plan, plan.nr - col);
        const int max_mr = nat::native_int8_call_rows(plan, cw / 16);
        if (nat::native_int8_group_span(plan, c.G, c.gs / 4, cw, rows, max_mr)
                >= c.G)
            return false;
        col += cw;
    }
    return true;
}

/// The call through native_brgemm_execute on its raw weight, packed for the
/// call only: matmul_direct's checks of the call do not run.
status_t run_direct(const nt::call_t &c) {
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda_or_tight();
    desc.ldb = c.ldb_or_tight();
    desc.ldc = c.ldc_or_tight();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = c.params.dtypes.src;
    desc.wei_dt = c.params.dtypes.wei;
    desc.dst_dt = c.params.dtypes.dst;
    desc.bias_dt = c.params.dtypes.bias;
    desc.bias = c.bias;
    desc.is_weights_const = true;
    desc.num_threads = c.params.num_threads;
    desc.gated_act = c.gated;
    nat::native_weight_ref_t ref;
    ref.weight = c.wei;
    ref.pack_nr = nat::kNativeLayoutPackNr;
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(
            desc, ref, c.src, c.dst, c.bias, c.params);
}

/// The reference algorithm on the call itself, without matmul_direct's
/// checks of the call.
status_t run_reference_direct(const nt::call_t &c) {
    mm::matmul_batch_params_t batch;
    mm::matmul_params p = c.params;
    p.lowoha_algo = matmul_algo_t::reference;
    return mm::reference_matmul_execute('r', c.transA, c.transB, c.M, c.N, c.K,
            c.alpha, c.src, c.lda_or_tight(), c.wei, c.ldb_or_tight(), c.bias,
            c.beta, c.dst, c.ldc_or_tight(), false, batch, p);
}

/// The call under ALGO 10 into a copy of the initial dst, calls times; the
/// native BRGEMM must serve every call, each with the same result.
buffer_t run_native(const gcase_t &c, const operands_t &o) {
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    buffer_t dst, first;
    for (int i = 0; i < c.calls; ++i) {
        dst = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        EXPECT_EQ(
                nt::run_matmul(make_call(c, o, o.wei.get(), dst.get(), scale)),
                status_t::success)
                << "call " << i;
        if (c.wc != 0) {
            EXPECT_GE(nt::native_layouts(o.wei.get()), 1u)
                    << "call " << i << " was not served by the native BRGEMM";
        }
        if (i == 0) {
            first = dst;
            continue;
        }
        EXPECT_TRUE(nt::within_steps(
                dst.get(), c.N, first.get(), c.N, c.dst, c.M, c.N, 0))
                << "call " << i << " vs call 0";
    }
    return dst;
}

bool vnni() {
    return nat::detect_uarch().avx512vnni;
}

class NativeBrgemmInt8PerGroup : public ::testing::TestWithParam<gcase_t> {
protected:
    void SetUp() override {
        if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
        mm::clear_matmul_weight_caches();
    }
    void TearDown() override { mm::clear_matmul_weight_caches(); }
    WeightCacheGuard global_cache_ {2};
};

TEST_P(NativeBrgemmInt8PerGroup, MatchesReferenceOrder) {
    const gcase_t &c = GetParam();
    const operands_t o = make_operands(c);
    if (c.batches && !tiles_take_batches(c, o))
        GTEST_SKIP() << "the plan of this CPU takes all groups of each tile "
                        "in one batch (their slab is within twice its pass "
                        "budget): no batches to test";
    const buffer_t raw = copy_of(o.wei);
    const source_t s = source_of(c, o);
    const buffer_t got = run_native(c, o);
    if (HasFailure()) return;

    // the reference order of the call
    const buffer_t model = model_dst(c, o, s);
    if (transcendental(c))
        EXPECT_TRUE(near_gelu(c, got.get(), model.get())) << "vs the model";
    else
        EXPECT_TRUE(nt::within_steps(
                got.get(), c.N, model.get(), c.N, c.dst, c.M, c.N, 0))
                << "vs the model of the reference order";

    // the reference algorithm, which takes a per-group source per group
    if (c.gran == gran_t::group) {
        buffer_t ref = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        nt::call_t rc = make_call(c, o, raw.get(), ref.get(), scale);
        rc.params.lowoha_algo = matmul_algo_t::reference;
        rc.params.weight_cache_type = 0;
        rc.params.num_threads = 0;
        ASSERT_EQ(nt::run_matmul(rc), status_t::success) << "reference";
        if (transcendental(c))
            EXPECT_TRUE(near_gelu(c, got.get(), ref.get()))
                    << "vs the reference algorithm";
        else
            EXPECT_TRUE(nt::within_steps(
                    got.get(), c.N, ref.get(), c.N, c.dst, c.M, c.N, 0))
                    << "vs the reference algorithm";
    }

    // f64 for the source scales the reference algorithm does not take per
    // group
    if (c.gran != gran_t::group && !transcendental(c) && c.po != po_t::add
            && !c.requant && !integer_dt(c.dst)) {
        size_t bad = 0;
        for (int m = 0; m < c.M; ++m)
            for (int n = 0; n < c.N; ++n) {
                double value = 0.0, bound = 0.0;
                f64_at(c, o, s, m, n, value, bound);
                if (c.dst == kBf16) bound += std::ldexp(std::fabs(value), -8);
                const float g = nt::load_f32(
                        got.get(), c.dst, static_cast<size_t>(m) * c.N + n);
                if (!(std::fabs(g - value) <= bound) && bad++ == 0)
                    ADD_FAILURE()
                            << "f64 at (" << m << ", " << n << "): got " << g
                            << ", f64 " << value << ", bound " << bound;
            }
        EXPECT_EQ(bad, 0u) << "elements outside the f64 bound";
    }

    // The default route sym-quantizes a per-token or per-group source scale
    // of more than one element, with one type for both scales, and its GEMM
    // takes alpha and beta as integers; a process-wide algorithm replaces it.
    if (src_scale_count(c) == 1 || c.src_scale_dt != c.wei_scale_dt
            || c.alpha != std::trunc(c.alpha) || c.beta != std::trunc(c.beta))
        return;
    if (!nt::default_route_reachable()) {
        RecordProperty("default", "not_run (process-wide algorithm)");
        return;
    }
    buffer_t def = copy_of(o.init);
    const buffer_t scale = copy_of(o.src_scale);
    nt::call_t dc = make_call(c, o, raw.get(), def.get(), scale);
    dc.params.weight_cache_type = 0;
    {
        const nt::algo_pin_t unset(matmul_algo_t::none);
        ASSERT_FALSE(nt::native_selected(dc.params))
                << "the default route is native";
        ASSERT_EQ(nt::run_matmul(dc), status_t::success) << "default route";
    }
    const bool f32 = c.dst == kF32;
    const float abs_bound
            = nt::comparison_abs_bound(c.K(), f32 ? epsilon_f32 : epsilon_bf16,
                    c.alpha, c.beta, !f32, true, false);
    EXPECT_TRUE(nt::within_band(got.get(), c.N, def.get(), c.N, c.dst, c.M, c.N,
            f32 ? rtol_f32 : rtol_bf16, abs_bound))
            << "vs the default route";
}

std::vector<gcase_t> named(std::vector<gcase_t> v) {
    std::vector<std::string> seen;
    for (gcase_t &c : v) {
        std::string n = describe(c);
        int dup = 0;
        for (const std::string &s : seen)
            dup += s == n || s.rfind(n + "_dup", 0) == 0;
        if (dup) n += "_dup" + std::to_string(dup);
        seen.push_back(n);
        c.name = n;
    }
    return v;
}

std::vector<gcase_t> cases() {
    const gcase_t base;
    std::vector<gcase_t> v;
    // group sizes (12, 48 and 96 not powers of two) over the source scales
    for (int gs : {4, 8, 12, 16, 32, 48, 64, 96, 128, 256})
        for (gran_t gran : {gran_t::group, gran_t::token, gran_t::tensor}) {
            gcase_t c = base;
            c.gs = gs;
            c.G = gs <= 12 ? 8 : 3;
            c.gran = gran;
            v.push_back(c);
        }
    // rows, columns
    for (int M : {1, 2, 7, 33, 128}) {
        gcase_t c = base;
        c.M = M;
        v.push_back(c);
    }
    for (int N : {32, 96, 4096})
        for (gran_t gran : {gran_t::group, gran_t::token}) {
            gcase_t c = base;
            c.N = N;
            c.M = N == 4096 ? 3 : 16;
            c.G = N == 4096 ? 2 : 4;
            c.gran = gran;
            v.push_back(c);
        }
    // N not a multiple of 32: the last column tile reads the scales of its
    // columns below N only; an s8 source with {M, G} scales, a u8 source with
    // a zero point, a source quantized in the call, alpha / beta, a bias, an
    // integer dst, several threads
    for (int N : {1, 24, 40, 100})
        for (int k = 0; k < 4; ++k) {
            gcase_t c = base;
            c.N = N;
            c.M = k == 0 ? 1 : 7 + 13 * k;
            c.threads = k % 2 ? 3 : 1;
            c.transB = k == 2;
            if (k == 1) {
                c.src = src_t::u8;
                c.gran = gran_t::tensor;
                c.zp = true;
            } else if (k == 2) {
                c.src = src_t::bf16;
                c.dst = kBf16;
                c.bias = kF32;
            } else if (k == 3) {
                c.gran = gran_t::tensor;
                c.dst = N % 2 ? kU8 : kS32;
                c.requant = c.dst == kU8;
                c.alpha = 0.5f;
                c.beta = 1.0f;
            }
            v.push_back(c);
        }
    // sources quantized in the call
    for (src_t src : {src_t::bf16, src_t::f32})
        for (gran_t gran : {gran_t::group, gran_t::token, gran_t::tensor})
            for (int M : {1, 7, 33}) {
                gcase_t c = base;
                c.src = src;
                c.gran = gran;
                c.M = M;
                c.gs = M == 7 ? 128 : 32;
                c.dst = src == src_t::bf16 ? kBf16 : kF32;
                v.push_back(c);
            }
    // transposed operands
    for (src_t src : {src_t::s8, src_t::bf16})
        for (gran_t gran : {gran_t::group, gran_t::token})
            for (int t = 1; t < 4; ++t) {
                gcase_t c = base;
                c.src = src;
                c.gran = gran;
                c.M = 13;
                c.transA = (t & 1) != 0;
                c.transB = (t & 2) != 0;
                v.push_back(c);
            }
    // destinations, biases, scale types, post-ops
    for (data_type_t dst : {kF32, kBf16})
        for (data_type_t bias : {kNone, kF32, kBf16}) {
            gcase_t c = base;
            c.M = 9;
            c.dst = dst;
            c.bias = bias;
            v.push_back(c);
        }
    for (data_type_t ws : {kBf16, kF16})
        for (data_type_t ss : {kF32, kBf16}) {
            gcase_t c = base;
            c.wei_scale_dt = ws;
            c.src_scale_dt = ss;
            c.gs = 64;
            v.push_back(c);
        }
    {
        gcase_t c = base;
        c.src = src_t::bf16;
        c.src_scale_dt = kBf16;
        c.wei_scale_dt = kBf16;
        c.dst = kBf16;
        v.push_back(c);
    }
    // many groups in one batch, the weight scales of each type read in
    // place
    for (data_type_t ws : {kF32, kBf16, kF16})
        for (data_type_t dst : {kF32, kBf16}) {
            gcase_t c = base;
            c.M = 13;
            c.gs = 8;
            c.G = 80;
            c.wei_scale_dt = ws;
            c.dst = dst;
            v.push_back(c);
        }
    // the weight scales of each type over several tiles and threads
    for (data_type_t ws : {kF32, kBf16, kF16}) {
        gcase_t c = base;
        c.M = 33;
        c.N = 320;
        c.gs = 16;
        c.G = 12;
        c.wei_scale_dt = ws;
        c.threads = 3;
        v.push_back(c);
    }
    for (po_t po : {po_t::add, po_t::gelu, po_t::gelu_add})
        for (data_type_t dst : {kF32, kBf16}) {
            gcase_t c = base;
            c.M = 7;
            c.po = po;
            c.dst = dst;
            c.bias = kBf16;
            v.push_back(c);
        }
    {
        gcase_t c = base;
        c.po = po_t::add;
        c.gran = gran_t::token;
        v.push_back(c);
    }
    // u8 / s8 / s32 destinations, with or without a dst scale and zero
    // point, over a per-tensor source scale (matmul_direct takes no other
    // with them): alpha and beta (reading the integer dst), a bias, a post-op,
    // a source quantized in the call, many groups, several tiles and threads
    for (data_type_t dst : {kU8, kS8, kS32})
        for (int k = 0; k < 6; ++k) {
            gcase_t c = base;
            c.gran = gran_t::tensor;
            c.dst = dst;
            c.requant = k % 2 == 0;
            c.M = 9;
            if (k == 1 || k == 2) {
                c.alpha = 0.5f;
                c.beta = 1.0f;
                c.bias = kF32;
            }
            if (k == 2) c.po = po_t::add;
            if (k == 3 && dst != kS32) {
                c.src = src_t::bf16;
                c.M = 7;
            }
            if (k == 4) {
                c.M = 13;
                c.gs = 8;
                c.G = 80;
                c.wei_scale_dt = kBf16;
                c.bias = kBf16;
            }
            if (k == 5) {
                c.M = 33;
                c.N = 320;
                c.gs = 16;
                c.G = 12;
                c.wei_scale_dt = kF16;
                c.threads = 3;
            }
            v.push_back(c);
        }
    // u8 sources, given with or without a zero point or a bf16 / f32 source
    // quantized to u8 in the call, over a per-tensor source scale
    // (matmul_direct takes no other with them): groups of several sizes,
    // zero points outside the byte range, transposed operands, the
    // destinations, a bias, alpha and beta, a post-op, many groups, several
    // tiles and threads, weight cache modes and repeated calls, and batches
    // of groups past twice the pass budget
    {
        gcase_t u8 = base;
        u8.src = src_t::u8;
        u8.gran = gran_t::tensor;
        u8.zp = true;
        for (int gs : {4, 8, 12, 32, 128})
            for (bool zp : {true, false}) {
                gcase_t c = u8;
                c.gs = gs;
                c.G = gs <= 12 ? 8 : 3;
                c.zp = zp;
                v.push_back(c);
            }
        for (int32_t zp_lo : {-2000, 1000}) {
            gcase_t c = u8;
            c.zp_lo = zp_lo;
            v.push_back(c);
        }
        // zero points whose group sums can pass the int32 range, given
        // (within and just past the bound of groups of 128, and far past
        // it) or made by the quantizer for bf16 / f32 sources of small
        // spread next to their offset
        for (int32_t zp_lo : {-130600, -130900, 1 << 24}) {
            gcase_t c = u8;
            c.gs = 128;
            c.G = 3;
            c.zp_lo = zp_lo;
            v.push_back(c);
        }
        for (src_t src : {src_t::bf16, src_t::f32}) {
            gcase_t c = u8;
            c.src = src;
            c.src_lo = 600.0f;
            c.src_hi = 601.0f;
            c.M = src == src_t::bf16 ? 7 : 33;
            c.gs = 128;
            c.G = 3;
            c.dst = src == src_t::bf16 ? kBf16 : kF32;
            c.threads = 3;
            v.push_back(c);
        }
        for (src_t src : {src_t::bf16, src_t::f32})
            for (int M : {1, 7, 33}) {
                gcase_t c = u8;
                c.src = src;
                c.M = M;
                c.gs = M == 7 ? 128 : 32;
                c.dst = src == src_t::bf16 ? kBf16 : kF32;
                v.push_back(c);
            }
        for (int t = 1; t < 4; ++t) {
            gcase_t c = u8;
            c.M = 13;
            c.transA = (t & 1) != 0;
            c.transB = (t & 2) != 0;
            v.push_back(c);
        }
        {
            gcase_t c = u8;
            c.src = src_t::bf16;
            c.M = 13;
            c.transA = true;
            c.dst = kBf16;
            v.push_back(c);
        }
        for (data_type_t dst : {kBf16, kU8, kS8, kS32}) {
            gcase_t c = u8;
            c.M = 9;
            c.dst = dst;
            c.requant = dst == kU8 || dst == kS8;
            c.bias = kF32;
            v.push_back(c);
        }
        {
            gcase_t c = u8;
            c.alpha = 2.0f;
            c.beta = 0.75f;
            c.bias = kBf16;
            c.po = po_t::add;
            v.push_back(c);
        }
        {
            gcase_t c = u8;
            c.M = 13;
            c.gs = 8;
            c.G = 80;
            c.wei_scale_dt = kF16;
            v.push_back(c);
        }
        {
            gcase_t c = u8;
            c.M = 33;
            c.N = 320;
            c.gs = 16;
            c.G = 12;
            c.wei_scale_dt = kBf16;
            c.threads = 3;
            v.push_back(c);
        }
        for (int threads : {1, 0}) {
            gcase_t c = u8;
            c.M = 128;
            c.N = 256;
            c.threads = threads;
            v.push_back(c);
        }
        for (int wc : {0, 1, 2}) {
            gcase_t c = u8;
            c.wc = wc;
            c.calls = 2;
            v.push_back(c);
        }
        {
            gcase_t c = u8;
            c.M = 40;
            c.N = 96;
            c.gs = 128;
            c.G = 57344 / 128;
            c.wei_scale_dt = kBf16;
            c.dst = kBf16;
            v.push_back(c);
        }
    }
    // alpha / beta after the last group, then the bias and the post-ops
    const float ab[][2] = {{0.5f, 0.0f}, {2.0f, 0.75f}, {1.0f, -1.0f}};
    for (const auto &x : ab)
        for (gran_t gran : {gran_t::group, gran_t::token}) {
            gcase_t c = base;
            c.alpha = x[0];
            c.beta = x[1];
            c.gran = gran;
            c.bias = kF32;
            c.po = gran == gran_t::group ? po_t::add : po_t::none;
            v.push_back(c);
        }
    {
        gcase_t c = base;
        c.alpha = 1.5f;
        c.beta = 0.5f;
        c.dst = kBf16;
        c.src = src_t::bf16;
        v.push_back(c);
    }
    // weight cache modes, repeated calls, thread counts
    for (int wc : {0, 1, 2}) {
        gcase_t c = base;
        c.wc = wc;
        c.calls = 2;
        v.push_back(c);
    }
    for (int threads : {1, 3, 0}) {
        gcase_t c = base;
        c.M = 128;
        c.N = 256;
        c.threads = threads;
        v.push_back(c);
    }
    // the shift sums claimed over several chunks: at K = 4096 a claim covers
    // 16 rows, so 100 rows take 7 claims, the last one partial, and tiles
    // wait on chunks other threads claimed
    for (int threads : {3, 0}) {
        gcase_t c = base;
        c.M = 100;
        c.N = 128;
        c.gs = 128;
        c.G = 32;
        c.threads = threads;
        v.push_back(c);
    }
    {
        gcase_t c = base;
        c.M = 128;
        c.N = 4096;
        c.gs = 128;
        c.G = 2;
        c.src = src_t::bf16;
        c.dst = kBf16;
        c.threads = 0;
        v.push_back(c);
    }
    // tiles of 40 rows, several row calls at 16 and at 32 columns, whose
    // slab of all K = 57344 passes twice the pass budget: the groups run in
    // batches, C carrying the sum from one to the next, stored to the
    // destination by the last one or left to the epilogue (skipped where
    // the plan of the CPU takes them in one batch)
    for (int gs : {32, 128})
        for (int N : {64, 96}) {
            gcase_t c = base;
            c.M = 40;
            c.N = N;
            c.gs = gs;
            c.G = 57344 / gs;
            c.wei_scale_dt = gs == 32 ? kF32 : (N == 64 ? kF16 : kBf16);
            c.dst = N == 64 ? kF32 : kBf16;
            c.bias = gs == 32 && N == 64 ? kF32 : kNone;
            c.batches = true;
            v.push_back(c);
        }
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8PerGroup, NativeBrgemmInt8PerGroup,
        ::testing::ValuesIn(cases()),
        [](const ::testing::TestParamInfo<gcase_t> &info) {
            return info.param.name;
        });

// A gated activation over [gate | up] halves, which the tiles pair (N 128):
// the group sums and scales of both halves in the reference order, then the
// activation within the band of the epilogue suite, over 4 and 40 groups of
// weight scales read in place.
TEST(NativeBrgemmInt8PerGroupGated, MatchesTheModel) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul})
        for (data_type_t dst : {kF32, kBf16})
            for (data_type_t ws : {kF32, kBf16, kF16})
                for (int G : {4, 40}) {
                    gcase_t c;
                    c.M = 7;
                    c.N = 128;
                    c.gs = G == 4 ? 32 : 8;
                    c.G = G;
                    c.dst = dst;
                    c.bias = kBf16;
                    c.wei_scale_dt = ws;
                    c.name = std::string("gated")
                            + std::to_string(static_cast<int>(act)) + "_"
                            + dt_name(dst) + "_ws" + dt_name(ws) + "_g"
                            + std::to_string(G);
                    SCOPED_TRACE(c.name);
                    const operands_t o = make_operands(c);
                    const source_t s = source_of(c, o);
                    const int I = c.N / 2;
                    buffer_t got = nt::make_buffer(
                            static_cast<size_t>(c.M) * I * nt::dt_bytes(dst));
                    const buffer_t scale = copy_of(o.src_scale);
                    nt::call_t call
                            = make_call(c, o, o.wei.get(), got.get(), scale);
                    call.gated = act;
                    call.ldc = I;
                    {
                        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
                        ASSERT_EQ(nt::run_matmul(call), status_t::success);
                        if (c.wc != 0) {
                            ASSERT_GE(nt::native_layouts(o.wei.get()), 1u)
                                    << "not served by the native BRGEMM";
                        }
                    }
                    std::vector<float> pre(static_cast<size_t>(c.M) * c.N);
                    for (int m = 0; m < c.M; ++m)
                        for (int n = 0; n < c.N; ++n)
                            pre[static_cast<size_t>(m) * c.N + n]
                                    = model_at(c, o, s, m, n);
                    std::vector<float> out(static_cast<size_t>(c.M) * I);
                    nt::gated_act_ref(pre.data(), c.M, c.N, act, out.data());
                    buffer_t want = nt::make_buffer(
                            static_cast<size_t>(c.M) * I * nt::dt_bytes(dst));
                    for (size_t i = 0; i < out.size(); ++i)
                        nt::store_f32(want.get(), dst, i, out[i]);
                    EXPECT_TRUE(nt::within_band(got.get(), I, want.get(), I,
                            dst, c.M, I,
                            dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f, 3e-4f));
                }
    mm::clear_matmul_weight_caches();
}

// Gated per-group calls through matmul_direct under ALGO 10 with beta (over
// the dst's N pre-activation columns), post-ops of the pre-activation, a u8
// source and requantized u8 / s8 destinations: the model of the reference
// order up to the pre-activation, then the activation within its band (one
// step for an integer dst); the columns past N / 2 keep the dst's values.
TEST(NativeBrgemmInt8PerGroupGated, BetaPostOpsAndDestinations) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul,
                 mm::matmul_gated_act_t::swiglu_oai_mul})
        for (int k = 0; k < 6; ++k) {
            gcase_t c;
            c.M = 7;
            c.N = 128;
            c.bias = kBf16;
            switch (k) {
                case 0: c.beta = 0.5f; break;
                case 1:
                    c.alpha = 2.0f;
                    c.beta = -1.0f;
                    c.dst = kBf16;
                    c.gs = 8;
                    c.G = 40;
                    break;
                case 2: c.po = po_t::gelu_add; break;
                case 3:
                    c.src = src_t::u8;
                    c.gran = gran_t::tensor;
                    c.zp = true;
                    c.beta = 0.25f;
                    c.po = po_t::add;
                    break;
                case 4:
                    c.gran = gran_t::tensor;
                    c.dst = kU8;
                    c.requant = true;
                    break;
                default:
                    c.src = src_t::bf16;
                    c.gran = gran_t::tensor;
                    c.dst = kS8;
                    c.requant = true;
                    c.beta = 1.0f;
                    break;
            }
            c.name = std::string("gated")
                    + std::to_string(static_cast<int>(act)) + "_" + describe(c);
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            const source_t s = source_of(c, o);
            const int I = c.N / 2;
            buffer_t got = copy_of(o.init);
            const buffer_t scale = copy_of(o.src_scale);
            nt::call_t call = make_call(c, o, o.wei.get(), got.get(), scale);
            call.gated = act;
            call.ldc = c.N;
            {
                const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
                ASSERT_EQ(nt::run_matmul(call), status_t::success);
                ASSERT_GE(nt::native_layouts(o.wei.get()), 1u)
                        << "not served by the native BRGEMM";
            }
            std::vector<float> pre(static_cast<size_t>(c.M) * c.N);
            for (int m = 0; m < c.M; ++m)
                for (int n = 0; n < c.N; ++n)
                    pre[static_cast<size_t>(m) * c.N + n]
                            = model_at(c, o, s, m, n);
            std::vector<float> out(static_cast<size_t>(c.M) * I);
            nt::gated_act_ref(pre.data(), c.M, c.N, act, out.data());
            buffer_t want = nt::make_buffer(
                    static_cast<size_t>(c.M) * I * nt::dt_bytes(c.dst));
            for (size_t i = 0; i < out.size(); ++i)
                store_ref(o, want.get(), c.dst, i, out[i]);
            if (integer_dt(c.dst))
                EXPECT_TRUE(nt::within_steps(
                        got.get(), c.N, want.get(), I, c.dst, c.M, I, 1));
            else
                EXPECT_TRUE(nt::within_band(got.get(), c.N, want.get(), I,
                        c.dst, c.M, I,
                        c.dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f, 3e-4f));
            for (int m = 0; m < c.M; ++m)
                for (int n = I; n < c.N; ++n) {
                    const size_t at = static_cast<size_t>(m) * c.N + n;
                    ASSERT_EQ(nt::load_f32(got.get(), c.dst, at),
                            nt::load_f32(o.init.get(), c.dst, at))
                            << "(" << m << ", " << n << ")";
                }
            mm::clear_matmul_weight_caches();
        }
}

// The batches of a per-group tile over a plain pack: all groups, unless the
// tile takes more than one row call and the slab of all groups plus one
// call's A rows passes twice the plan's pass budget; then the fewest equal
// batches that fit the budget, or one group per batch when no group fits.
TEST(NativeBrgemmInt8GroupTile, SpanFollowsThePassBudget) {
    size_t bad = 0;
    nat::native_plan_t plan;
    for (int64_t budget : {0, 4863, 4864, 1 << 19, 1 << 20})
        for (int groups : {1, 2, 7, 32, 112, 224, 448, 896})
            for (int group_quads : {2, 8, 32})
                for (int cw : {16, 32})
                    for (int max_mr : {6, 12})
                        for (int rows : {1, max_mr, max_mr + 1, 40, 192}) {
                            plan.int8_pass_budget = budget;
                            const int got = nat::native_int8_group_span(plan,
                                    groups, group_quads, cw, rows, max_mr);
                            const int64_t group_bytes
                                    = int64_t {4} * group_quads * (cw + max_mr);
                            int want = groups;
                            if (rows > max_mr
                                    && groups * group_bytes > 2 * budget)
                                for (int b = 1; b <= groups; ++b) {
                                    want = (groups + b - 1) / b;
                                    if (want * group_bytes <= budget) break;
                                }
                            if (got != want && bad++ == 0)
                                ADD_FAILURE()
                                        << "budget " << budget << " groups "
                                        << groups << " group_quads "
                                        << group_quads << " cw " << cw
                                        << " rows " << rows << " max_mr "
                                        << max_mr << ": " << got << ", want "
                                        << want;
                        }
    EXPECT_EQ(bad, 0u);
}

// Settings of the plan's INT8 keys change how a tile is cut, not the order
// of any output element's operations: the same bits as the built-in plan.
TEST(NativeBrgemmInt8PerGroupPlan, PinsKeepTheBits) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    if (!nt::in_child()) {
        for (const char *name : {"ZENDNNL_MATMUL_NATIVE_CONFIG",
                     "ZENDNNL_MATMUL_NATIVE_TUNING_FILE"}) {
            const char *value = std::getenv(name);
            if (value != nullptr && *value != '\0')
                GTEST_SKIP() << name << " is set in this process";
        }
    }
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    // Settings of per-channel keys only are not applied to a per-group call.
    const std::pair<const char *, bool> settings[] = {{"int8_kc=64", false},
            {"int8_kc=4096", false}, {"l1_pass=0", false}, {"int8_mr=5", true},
            {"int8_cw=16", true}, {"int8_cw=32/16,int8_mr=3", true},
            {"nr=16", true}, {"mb=8", true}, {"bias_in_dequant=0", false},
            {"direct_store=0", true}, {"threads_cap=1", true},
            {"schedule=row_groups,domain_threads=2", true},
            {"int8_kc=64,int8_mr=5", true}, {"schedule=blocks", true},
            {"schedule=blocks,unit_tiles=3,domain_threads=2", true},
            {"schedule=blocks,unit_tiles=2,domain_threads=2", true}};
    std::vector<gcase_t> v;
    {
        gcase_t c;
        c.M = 40;
        c.N = 128;
        c.gs = 32;
        c.G = 8;
        c.bias = kF32;
        v.push_back(c);
        c.src = src_t::bf16;
        c.gran = gran_t::token;
        c.dst = kBf16;
        c.alpha = 0.5f;
        c.beta = 0.25f;
        v.push_back(c);
        // Several shift claims (K = 4096), under every setting; its own M
        // range keeps its override clauses apart from the others'.
        c = gcase_t {};
        c.M = 100;
        c.N = 128;
        c.gs = 128;
        c.G = 32;
        v.push_back(c);
    }
    // The process planner reads its settings once: one child process runs
    // setting j on each case at M + j, through an override clause per call.
    std::vector<gcase_t> calls;
    std::vector<size_t> setting_of;
    std::string config;
    for (const gcase_t &base : named(v))
        for (size_t j = 0; j < std::size(settings); ++j) {
            gcase_t c = base;
            c.M += static_cast<int>(j);
            calls.push_back(c);
            setting_of.push_back(j);
            config += "m=" + std::to_string(c.M)
                    + ",src=" + (c.src == src_t::s8 ? "s8" : "bf16") + ":"
                    + settings[j].first + ";";
        }
    std::vector<operands_t> ops;
    for (const gcase_t &c : calls)
        ops.push_back(make_operands(c));
    nt::child_buffers_t shared;
    if (!nt::in_child()) {
        std::vector<buffer_t> refs;
        for (size_t i = 0; i < calls.size(); ++i)
            refs.push_back(run_native(calls[i], ops[i]));
        shared.publish(refs);
        const std::unique_ptr<nat::native_planner_t> planner
                = nat::make_native_planner(config.c_str(), nullptr);
        for (size_t i = 0; i < calls.size(); ++i) {
            SCOPED_TRACE(calls[i].name + " M=" + std::to_string(calls[i].M)
                    + " " + settings[setting_of[i]].first);
            const buffer_t scale = copy_of(ops[i].src_scale);
            const nat::native_plan_t plan = planner->plan(problem_of(make_call(
                    calls[i], ops[i], ops[i].wei.get(), nullptr, scale)));
            EXPECT_EQ(nt::top_source(plan),
                    settings[setting_of[i]].second
                            ? nat::native_plan_source_t::override
                            : nat::native_plan_source_t::heuristic);
            EXPECT_EQ(nat::native_plan_config_text(plan).find("int8_kc"),
                    std::string::npos);
        }
    }
    nt::expect_in_child({{"ZENDNNL_MATMUL_NATIVE_CONFIG", config.c_str()},
                                {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
            [&] {
        const std::vector<buffer_t> &refs = shared.buffers();
        if (refs.size() != calls.size()) {
            std::fprintf(stderr, "the parent's outputs are missing\n");
            return 1;
        }
        int bad = 0;
        for (size_t i = 0; i < calls.size(); ++i) {
            const gcase_t &c = calls[i];
            const buffer_t got = run_native(c, ops[i]);
            if (!nt::within_steps(got.get(), c.N, refs[i].get(), c.N, c.dst,
                        c.M, c.N, 0)) {
                std::fprintf(stderr, "%s M=%d with %s differs\n",
                        c.name.c_str(), c.M, settings[setting_of[i]].first);
                ++bad;
            }
        }
        return bad + (::testing::Test::HasFailure() ? 1 : 0);
    }, "per-group calls under per-call settings");
    mm::clear_matmul_weight_caches();
}

// The tile check: what a group-scaled call needs, and the problem class of a
// group-scaled call with a bias (the epilogue adds it).
TEST(NativeBrgemmInt8PerGroupCheck, ServedCalls) {
    const int M = 4, N = 64, K = 128;
    std::vector<float> scales(static_cast<size_t>(K) * N, 0.01f);
    const auto call = [&](data_type_t src, int groups) {
        mm::matmul_params p;
        p.dtypes.src = src;
        p.dtypes.wei = kS8;
        p.dtypes.dst = kBf16;
        set_quant(p.quant_params.wei_scale, scales.data(), kF32, {groups, N});
        set_quant(p.quant_params.src_scale, scales.data(), kF32, {M, groups});
        return p;
    };
    nat::GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = K;
    desc.ldb = N;
    desc.ldc = N;
    desc.src_dt = kS8;
    desc.wei_dt = kS8;
    desc.dst_dt = kBf16;
    const auto check = [&](const mm::matmul_params &p) {
        nat::GemmDescriptor d = desc;
        d.src_dt = p.dtypes.src;
        d.dst_dt = p.dtypes.dst;
        return nat::check_int8_call(d, p, nat::decide_int8_call(d, p, nullptr));
    };
    EXPECT_EQ(nat::int8_scale_groups(call(kS8, 4), N), 4);
    EXPECT_EQ(nat::int8_scale_groups(call(kS8, 1), N), 0);
    EXPECT_EQ(check(call(kS8, 4)), status_t::success);
    EXPECT_EQ(check(call(kS8, 32)), status_t::success) << "groups of 4";
    EXPECT_EQ(check(call(kS8, 64)), status_t::unimplemented) << "groups of 2";
    EXPECT_EQ(check(call(kS8, 3)), status_t::unimplemented) << "K % G";
    auto p = call(kS8, 4);
    p.quant_params.src_scale.dims = {M, 8};
    EXPECT_EQ(check(p), status_t::unimplemented) << "a source G other than G";
    p.quant_params.src_scale.dims = {M, 1};
    EXPECT_EQ(check(p), status_t::success) << "per token";
    p.quant_params.src_scale.dims = {1, 1};
    EXPECT_EQ(check(p), status_t::success) << "per tensor";
    p.quant_params.src_scale.buff = nullptr;
    EXPECT_EQ(check(p), status_t::success) << "no source scale";
    p = call(kU8, 4);
    EXPECT_EQ(check(p), status_t::unimplemented) << "a u8 source per group";
    p.quant_params.src_scale.dims = {M, 1};
    EXPECT_EQ(check(p), status_t::unimplemented) << "a u8 source per token";
    p.quant_params.src_scale.dims = {1, 1};
    EXPECT_EQ(check(p), status_t::success) << "a u8 source";
    int32_t zp = 0;
    for (data_type_t zp_dt : {kS32, kS8, kU8}) {
        set_quant(p.quant_params.src_zp, &zp, zp_dt, {1, 1});
        EXPECT_EQ(check(p), status_t::success)
                << "a u8 source with a zero point";
    }
    set_quant(p.quant_params.src_zp, &zp, kF32, {1, 1});
    EXPECT_EQ(check(p), status_t::unimplemented) << "an f32 zero point";
    set_quant(p.quant_params.src_zp, &zp, kS32, {M, 1});
    EXPECT_EQ(check(p), status_t::unimplemented) << "a zero point per row";
    p = call(kS8, 4);
    set_quant(p.quant_params.src_zp, &zp, kS32, {1, 1});
    EXPECT_EQ(check(p), status_t::unimplemented) << "an s8 source zero point";
    for (data_type_t dst : {kU8, kS8, kS32, kF16}) {
        p = call(kS8, 4);
        p.dtypes.dst = dst;
        EXPECT_EQ(check(p),
                dst == kF16 ? status_t::unimplemented : status_t::success)
                << dt_name(dst);
    }
    p = call(kS8, 4);
    p.quant_params.wei_scale.dt = kS32;
    EXPECT_EQ(check(p), status_t::unimplemented) << "an s32 weight scale";
    p = call(kBf16, 4);
    EXPECT_EQ(check(p), status_t::unimplemented) << "bf16 without quantization";
    p.dynamic_quant = true;
    p.dtypes.compute = kS8;
    EXPECT_EQ(check(p), status_t::success) << "bf16 quantized to s8";
    p.dtypes.compute = kU8;
    EXPECT_EQ(check(p), status_t::unimplemented)
            << "bf16 quantized to u8 per group";
    p.quant_params.src_scale.dims = {1, 1};
    EXPECT_EQ(check(p), status_t::success) << "bf16 quantized to u8";
    set_quant(p.quant_params.src_zp, nullptr, kS32, {M, 1});
    EXPECT_EQ(check(p), status_t::unimplemented)
            << "bf16 quantized to u8 with zero points per row";

    std::vector<float> bias(N, 0.5f);
    nt::call_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.bias = bias.data();
    c.params = call(kS8, 4);
    c.params.dtypes.bias = kF32;
    EXPECT_EQ(problem_of(c).epilogue, nat::native_epilogue_class_t::bias);
    c.params.quant_params.wei_scale.dims = {1, N};
    c.params.quant_params.src_scale.dims = {M, 1};
    EXPECT_EQ(problem_of(c).epilogue, nat::native_epilogue_class_t::store);
}

// Calls the native BRGEMM does not serve return unimplemented with
// lowoha_algo native_brgemm, with dst untouched and no layout of their
// weight: groups of other than whole K quads, and an s8 source with a zero
// point.
TEST(NativeBrgemmInt8PerGroupDecline, UnservedCallsReturnUnimplemented) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    struct decline_t {
        const char *name;
        gcase_t c;
        bool src_zp;
    };
    std::vector<decline_t> v;
    {
        gcase_t c;
        c.gs = 6;
        c.G = 8;
        v.push_back({"group_size_6", c, false});
        c.gs = 10;
        c.G = 4;
        v.push_back({"group_size_10", c, false});
    }
    {
        gcase_t c;
        c.gran = gran_t::tensor;
        v.push_back({"s8_source_zero_point", c, true});
    }
    for (decline_t &d : v) {
        SCOPED_TRACE(d.name);
        d.c.name = d.name;
        operands_t o = make_operands(d.c);
        int32_t zp = 128;
        if (d.src_zp)
            set_quant(o.params.quant_params.src_zp, &zp, kS32, {1, 1});
        buffer_t dst = copy_of(o.init);
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        nt::call_t call
                = make_call(d.c, o, o.wei.get(), dst.get(), o.src_scale);
        call.params.lowoha_algo = matmul_algo_t::native_brgemm;
        EXPECT_TRUE(nt::refused_with(call, status_t::unimplemented));
        EXPECT_EQ(nt::native_layouts(o.wei.get()), 0u)
                << "served by the native BRGEMM";
    }
    // matmul_direct's checks refuse a per-group source with per-channel
    // weight scales on every route; the tile serves it (see
    // NativeBrgemmInt8SourceGroups).
    if (zendnnl::common::op_instrumentation::is_enabled()) {
        gcase_t c;
        c.name = "group_source_channel_weight";
        c.wgran = wgran_t::channel;
        operands_t o = make_operands(c);
        buffer_t dst = copy_of(o.init);
        const nt::call_t call
                = make_call(c, o, o.wei.get(), dst.get(), o.src_scale);
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        EXPECT_NE(nt::run_matmul(call), status_t::success);
        EXPECT_EQ(nt::native_layouts(o.wei.get()), 0u);
        nat::GemmDescriptor desc;
        desc.M = c.M;
        desc.N = c.N;
        desc.K = c.K();
        desc.dst_dt = c.dst;
        EXPECT_EQ(nat::check_int8_call(desc, call.params,
                          nat::decide_int8_call(desc, call.params, nullptr)),
                status_t::success);
    }
    mm::clear_matmul_weight_caches();
}

// The reference algorithm takes a per-group {M, G} source scale group by
// group, which matmul_direct passes on only with an f32 or bf16 dst; both
// run here on the call itself: u8 / s8 / s32 dsts, requantized or not, with
// alpha, beta (reading the integer dst), a bias and a post-op, bit for bit.
TEST(NativeBrgemmInt8PerGroupRequant, MatchesTheReferenceAlgorithm) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    for (data_type_t dst : {kU8, kS8, kS32})
        for (int k = 0; k < 4; ++k) {
            gcase_t c;
            c.M = 11;
            c.dst = dst;
            c.requant = k != 1;
            if (k >= 2) {
                c.alpha = 0.5f;
                c.beta = 1.0f;
                c.bias = kF32;
            }
            if (k == 3) {
                c.po = po_t::add;
                c.gs = 8;
                c.G = 40;
                c.wei_scale_dt = kBf16;
                c.src_scale_dt = kBf16;
            }
            c.wc = 0;
            c.name = describe(c);
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            buffer_t got = copy_of(o.init), want = copy_of(o.init);
            nt::call_t call
                    = make_call(c, o, o.wei.get(), got.get(), o.src_scale);
            ASSERT_EQ(run_direct(call), status_t::success);
            call.dst = want.get();
            ASSERT_EQ(run_reference_direct(call), status_t::success);
            EXPECT_TRUE(nt::within_steps(
                    got.get(), c.N, want.get(), c.N, c.dst, c.M, c.N, 0));
            const source_t s = source_of(c, o);
            const buffer_t model = model_dst(c, o, s);
            EXPECT_TRUE(nt::within_steps(
                    got.get(), c.N, model.get(), c.N, c.dst, c.M, c.N, 0))
                    << "vs the model of the reference order";
        }
}

std::vector<gcase_t> source_group_cases() {
    gcase_t base;
    base.wgran = wgran_t::channel;
    std::vector<gcase_t> v;
    // weight scales and destinations
    for (wgran_t wgran : {wgran_t::channel, wgran_t::tensor, wgran_t::none})
        for (data_type_t dst : {kF32, kBf16, kU8, kS8, kS32}) {
            gcase_t c = base;
            c.M = 9;
            c.wgran = wgran;
            c.dst = dst;
            c.requant = dst == kU8 || dst == kS8;
            v.push_back(c);
        }
    // group sizes (12 not a power of two)
    for (int gs : {4, 12, 64, 256}) {
        gcase_t c = base;
        c.gs = gs;
        c.G = gs <= 12 ? 8 : 3;
        v.push_back(c);
    }
    // alpha before the weight scales, beta after them into f32 and before
    // them into s8, then the bias and a post-op
    const float ab[][2] = {{0.5f, 0.0f}, {2.0f, 0.75f}, {1.0f, -1.0f}};
    for (const auto &x : ab)
        for (data_type_t dst : {kF32, kS8}) {
            gcase_t c = base;
            c.alpha = x[0];
            c.beta = x[1];
            c.bias = kF32;
            c.po = po_t::add;
            c.dst = dst;
            c.requant = dst == kS8;
            v.push_back(c);
        }
    // scale types, transposed operands, tiles and threads
    for (data_type_t ws : {kBf16, kF16}) {
        gcase_t c = base;
        c.wei_scale_dt = ws;
        c.src_scale_dt = kBf16;
        c.bias = kBf16;
        v.push_back(c);
    }
    for (int t = 1; t < 4; ++t) {
        gcase_t c = base;
        c.M = 13;
        c.transA = (t & 1) != 0;
        c.transB = (t & 2) != 0;
        v.push_back(c);
    }
    {
        gcase_t c = base;
        c.M = 33;
        c.N = 320;
        c.gs = 16;
        c.G = 12;
        c.threads = 3;
        v.push_back(c);
        c.M = 128;
        c.N = 256;
        c.gs = 32;
        c.G = 4;
        c.threads = 0;
        v.push_back(c);
    }
    // sources quantized per group in the call
    for (src_t src : {src_t::bf16, src_t::f32})
        for (int M : {1, 7, 33}) {
            gcase_t c = base;
            c.src = src;
            c.M = M;
            c.gs = M == 7 ? 128 : 32;
            c.dst = src == src_t::bf16 ? kBf16 : kF32;
            c.wgran = M == 7 ? wgran_t::tensor : wgran_t::channel;
            v.push_back(c);
        }
    // batches of groups past twice the pass budget, C carrying the sum
    {
        gcase_t c = base;
        c.M = 40;
        c.N = 96;
        c.gs = 128;
        c.G = 57344 / 128;
        c.dst = kBf16;
        v.push_back(c);
    }
    for (gcase_t &c : v)
        c.wc = 0;
    return named(v);
}

// A per-group {M, G} source scale with per-tensor or per-channel weight
// scales, or none: the reference algorithm adds f32(sum) * src scale over
// the groups of each row, then applies alpha and the weight scale, with
// beta after the weight scale into an f32 or bf16 dst and before it into
// any other.
// matmul_direct's checks refuse the pairing on every route, so the native
// BRGEMM runs on the call itself: against the reference algorithm on the
// same call bit for bit (a static s8 source), and against the model of that
// order (a source quantized in the call too).
TEST(NativeBrgemmInt8SourceGroups, MatchesTheReferenceAlgorithm) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    for (const gcase_t &c : source_group_cases()) {
        SCOPED_TRACE(c.name);
        const operands_t o = make_operands(c);
        const source_t s = source_of(c, o);
        buffer_t got = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        nt::call_t call = make_call(c, o, o.wei.get(), got.get(), scale);
        ASSERT_EQ(run_direct(call), status_t::success);
        const buffer_t model = model_dst(c, o, s);
        EXPECT_TRUE(nt::within_steps(
                got.get(), c.N, model.get(), c.N, c.dst, c.M, c.N, 0))
                << "vs the model of the reference order";
        if (c.src != src_t::s8) continue;
        buffer_t want = copy_of(o.init);
        call.dst = want.get();
        ASSERT_EQ(run_reference_direct(call), status_t::success);
        EXPECT_TRUE(nt::within_steps(
                got.get(), c.N, want.get(), c.N, c.dst, c.M, c.N, 0))
                << "vs the reference algorithm";
    }
}

// The same calls with a gated activation and beta over the dst's N
// pre-activation columns (the gate and up columns a paired tile holds, or
// the interleaved pairs of swiglu_oai_mul): the model of the reference
// order up to the pre-activation, then the activation within its band.
TEST(NativeBrgemmInt8SourceGroups, GatedMatchesTheModel) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::swiglu_oai_mul})
        for (wgran_t wgran : {wgran_t::channel, wgran_t::tensor}) {
            gcase_t c;
            c.M = 9;
            c.N = 128;
            c.wgran = wgran;
            c.alpha = 1.5f;
            c.beta = 0.5f;
            c.bias = kF32;
            c.dst = wgran == wgran_t::channel ? kF32 : kBf16;
            c.wc = 0;
            c.name = std::string("gated")
                    + std::to_string(static_cast<int>(act)) + "_" + describe(c);
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            const source_t s = source_of(c, o);
            const int I = c.N / 2;
            buffer_t got = copy_of(o.init);
            const buffer_t scale = copy_of(o.src_scale);
            nt::call_t call = make_call(c, o, o.wei.get(), got.get(), scale);
            call.gated = act;
            call.ldc = c.N;
            ASSERT_EQ(run_direct(call), status_t::success);
            std::vector<float> pre(static_cast<size_t>(c.M) * c.N);
            for (int m = 0; m < c.M; ++m)
                for (int n = 0; n < c.N; ++n)
                    pre[static_cast<size_t>(m) * c.N + n]
                            = model_at(c, o, s, m, n);
            std::vector<float> out(static_cast<size_t>(c.M) * I);
            nt::gated_act_ref(pre.data(), c.M, c.N, act, out.data());
            buffer_t want = nt::make_buffer(
                    static_cast<size_t>(c.M) * I * nt::dt_bytes(c.dst));
            for (size_t i = 0; i < out.size(); ++i)
                nt::store_f32(want.get(), c.dst, i, out[i]);
            EXPECT_TRUE(nt::within_band(got.get(), c.N, want.get(), I, c.dst,
                    c.M, I, c.dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f,
                    3e-4f));
        }
}

// Without matmul_direct's checks (ZENDNNL_DIAGNOSTICS_ENABLE=0) such a call
// reaches the routes, and the native BRGEMM serves it under ALGO 10 with the
// bits of the call on its own.
TEST(NativeBrgemmInt8SourceGroups, ServedWithoutTheChecks) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    std::vector<gcase_t> v;
    {
        gcase_t c;
        c.wgran = wgran_t::channel;
        c.M = 13;
        v.push_back(c);
        c.src = src_t::bf16;
        c.dst = kBf16;
        c.wgran = wgran_t::tensor;
        v.push_back(c);
        c = gcase_t {};
        c.wgran = wgran_t::channel;
        c.dst = kS8;
        c.requant = true;
        c.alpha = 0.5f;
        c.beta = 1.0f;
        c.bias = kF32;
        v.push_back(c);
    }
    v = named(v);
    std::vector<operands_t> ops;
    for (const gcase_t &c : v)
        ops.push_back(make_operands(c));
    nt::child_buffers_t shared;
    if (!nt::in_child()) {
        std::vector<buffer_t> refs;
        for (size_t i = 0; i < v.size(); ++i) {
            buffer_t dst = copy_of(ops[i].init);
            const buffer_t scale = copy_of(ops[i].src_scale);
            nt::call_t call = make_call(
                    v[i], ops[i], ops[i].wei.get(), dst.get(), scale);
            call.params.weight_cache_type = 0;
            ASSERT_EQ(run_direct(call), status_t::success) << v[i].name;
            refs.push_back(dst);
        }
        shared.publish(refs);
    }
    nt::expect_in_child({{"ZENDNNL_DIAGNOSTICS_ENABLE", "0"}}, [&] {
        const std::vector<buffer_t> &refs = shared.buffers();
        if (refs.size() != v.size()) {
            std::fprintf(stderr, "the parent's outputs are missing\n");
            return 1;
        }
        int bad = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            const gcase_t &c = v[i];
            const buffer_t got = run_native(c, ops[i]);
            if (!nt::within_steps(got.get(), c.N, refs[i].get(), c.N, c.dst,
                        c.M, c.N, 0)) {
                std::fprintf(stderr, "%s differs\n", c.name.c_str());
                ++bad;
            }
        }
        return bad + (::testing::Test::HasFailure() ? 1 : 0);
    }, "per-group source scales with other weight scales, unchecked");
    mm::clear_matmul_weight_caches();
}

// The dims of a null source-scale buffer scale nothing: a static s8 call
// with per-channel weight scales and no source scale, whose dims still read
// {M, G} (dt none or f32), keeps the per-channel decision, and matmul_direct
// serves it under ALGO 10 with the bits of the same call without those
// dims.  A source the call quantizes takes its groups from the dims, its
// quantizer making the scales.
TEST(NativeBrgemmInt8SourceGroups, DimsWithoutAScaleKeepThePerChannelCall) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    gcase_t c;
    c.wgran = wgran_t::channel;
    c.name = describe(c);
    const operands_t o = make_operands(c);
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K();
    desc.lda = c.K();
    desc.ldb = c.N;
    desc.ldc = c.N;
    desc.src_dt = kS8;
    desc.wei_dt = kS8;
    desc.dst_dt = c.dst;
    const auto without_scale
            = [&](data_type_t dt, std::vector<int64_t> dims, void *dst) {
        nt::call_t call = make_call(c, o, o.wei.get(), dst, o.src_scale);
        auto &sc = call.params.quant_params.src_scale;
        sc.buff = nullptr;
        sc.dt = dt;
        sc.dims = std::move(dims);
        return call;
    };
    buffer_t want = copy_of(o.init);
    {
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        ASSERT_EQ(nt::run_matmul(without_scale(kNone, {}, want.get())),
                status_t::success);
    }
    for (const data_type_t dt : {kNone, kF32}) {
        SCOPED_TRACE(dt_name(dt));
        buffer_t got = copy_of(o.init);
        const nt::call_t call = without_scale(dt, {c.M, c.G}, got.get());
        EXPECT_EQ(nat::int8_source_groups(call.params, c.M, c.N), 0);
        const nat::native_int8_decision_t d
                = nat::decide_int8_call(desc, call.params, nullptr);
        EXPECT_EQ(d.scheme, nat::native_int8_scheme_t::per_channel);
        EXPECT_FALSE(d.scales_after_groups);
        EXPECT_EQ(
                nat::check_int8_call(desc, call.params, d), status_t::success);
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        ASSERT_EQ(nt::run_matmul(call), status_t::success);
        EXPECT_TRUE(nt::within_steps(
                got.get(), c.N, want.get(), c.N, c.dst, c.M, c.N, 0));
    }
    mm::matmul_params q = without_scale(kF32, {c.M, c.G}, nullptr).params;
    q.dtypes.src = kBf16;
    q.dynamic_quant = true;
    q.dtypes.compute = kS8;
    EXPECT_EQ(nat::int8_source_groups(q, c.M, c.N), c.G);
    mm::clear_matmul_weight_caches();
}

TEST(NativeBrgemmInt8GroupKernel, Table) {
    for (int nv : {1, 2}) {
        const int max_mr = nat::native_int8_group_ukernel_max_mr(nv);
        EXPECT_GE(max_mr, 6) << nv;
        EXPECT_LE(max_mr, 12) << nv;
        for (bool src_u8 : {false, true}) {
            EXPECT_EQ(nat::select_native_int8_group_ukernel(0, nv, src_u8),
                    nullptr);
            EXPECT_EQ(nat::select_native_int8_group_ukernel(
                              max_mr + 1, nv, src_u8),
                    nullptr);
            for (int mr = 1; mr <= max_mr; ++mr)
                EXPECT_NE(nat::select_native_int8_group_ukernel(mr, nv, src_u8),
                        nullptr)
                        << mr << " " << nv << " " << src_u8;
        }
        // The wide form of a u8 source: its own kernels, of the same rows.
        EXPECT_EQ(nat::select_native_int8_group_ukernel(0, nv, true, true),
                nullptr);
        EXPECT_EQ(nat::select_native_int8_group_ukernel(
                          max_mr + 1, nv, true, true),
                nullptr);
        for (int mr = 1; mr <= max_mr; ++mr) {
            const auto wide
                    = nat::select_native_int8_group_ukernel(mr, nv, true, true);
            EXPECT_NE(wide, nullptr) << mr << " " << nv;
            EXPECT_NE(wide, nat::select_native_int8_group_ukernel(mr, nv, true))
                    << mr << " " << nv;
        }
    }
    for (int nv : {3, 4}) {
        EXPECT_EQ(nat::native_int8_group_ukernel_max_mr(nv), 0);
        for (bool src_u8 : {false, true})
            EXPECT_EQ(nat::select_native_int8_group_ukernel(1, nv, src_u8),
                    nullptr);
    }
}

// The batch-reduce contract of the group kernel on its own: per element the
// exact int32 sum (of an s8 source, its shift plus the weight read as w +
// 128; of a u8 source, the sum of (a - zero point) * w, exact in the wide
// form for zero points far outside the byte range), then source and weight
// scales (f32, bf16 or f16) and one add per element, in batch order from C
// (or +0.0), stored to C or to a bf16 / f32 destination; bit for bit
// against a scalar model, over both sources, every row count, width and
// weight scale type, with the store, the entry C, per-group source scales,
// the source padding and the zero point drawn from a generator of their
// own, independently of each other and of the shapes.  Nothing past the
// tile's columns is written, and C is not written when the tile stores to
// a destination.
TEST(NativeBrgemmInt8GroupKernel, MatchesTheModel) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    enum class store_t { c, f32, bf16 };
    std::mt19937 rng(20261026);
    std::uniform_int_distribution<int> byte(-128, 127);
    std::uniform_real_distribution<float> scale(0.002f, 0.01f);
    std::mt19937 draws(7);
    std::uniform_int_distribution<int> pick(0, 35);
    std::uniform_int_distribution<int> zero_point(-300, 600);
    std::uniform_int_distribution<int> wide_zero_point(-(1 << 24), 1 << 24);
    struct kind_t {
        data_type_t sb_dt;
        bool src_u8, wide;
    };
    // The s8 source, the u8 source, then the u8 source's wide form, with
    // each weight scale type.
    const kind_t kinds[] = {{kF32, false, false}, {kBf16, false, false},
            {kF16, false, false}, {kF32, true, false}, {kBf16, true, false},
            {kF16, true, false}, {kF32, true, true}, {kBf16, true, true},
            {kF16, true, true}};
    int variant = 0;
    for (const auto &[sb_dt, u8_kind, wide] : kinds)
        for (int nv : {1, 2})
            for (int mr = 1; mr <= nat::native_int8_group_ukernel_max_mr(nv);
                    ++mr)
                for (int k_quads : {1, 3, 8, 32})
                    for (int batch : {1, 2, 5}) {
                        const int v = variant++;
                        const int nr = 16 * nv;
                        const int pack_nr = 64, col0 = nv == 1 ? 16 : 32;
                        const int K = batch * k_quads * 4;
                        const int s = pick(draws);
                        const store_t store = static_cast<store_t>(s % 3);
                        const bool accumulate = (s / 3) % 2 == 1;
                        const bool per_group_src = (s / 6) % 2 == 0;
                        const int lda = K + 4 * ((s / 12) % 3);
                        // A zero sum with a negative scale gives -0.0, which
                        // the first element adds to +0.0.
                        const bool zero_row = v % 5 == 2;
                        const bool src_u8 = u8_kind;
                        const int32_t zp = !src_u8 ? 0
                                : wide             ? wide_zero_point(draws)
                                : v % 3 == 0       ? 0
                                : v % 3 == 1       ? 128
                                                   : zero_point(draws);
                        SCOPED_TRACE("mr=" + std::to_string(mr)
                                + " nv=" + std::to_string(nv)
                                + " k_quads=" + std::to_string(k_quads)
                                + " batch=" + std::to_string(batch) + " store="
                                + std::to_string(static_cast<int>(store))
                                + " accumulate=" + std::to_string(accumulate)
                                + " per_group_src="
                                + std::to_string(per_group_src)
                                + " lda=" + std::to_string(lda)
                                + " wei_scale=" + dt_name(sb_dt)
                                + " src_u8=" + std::to_string(src_u8)
                                + " wide=" + std::to_string(wide)
                                + " zp=" + std::to_string(zp));
                        std::vector<int8_t> a(static_cast<size_t>(mr) * lda);
                        for (auto &x : a)
                            x = static_cast<int8_t>(byte(rng));
                        if (zero_row)
                            std::fill(a.begin(), a.begin() + lda, int8_t {0});
                        // The source value at (m, k) as the kernel reads it.
                        const auto a_at = [&](int m, int k) {
                            const int8_t x
                                    = a[static_cast<size_t>(m) * lda + k];
                            return src_u8
                                    ? static_cast<int>(static_cast<uint8_t>(x))
                                    : static_cast<int>(x);
                        };
                        const int quads = batch * k_quads;
                        std::vector<int8_t> b(
                                static_cast<size_t>(quads) * pack_nr * 4);
                        for (auto &x : b)
                            x = static_cast<int8_t>(byte(rng));
                        const auto w_at = [&](int k, int n) {
                            return b[(static_cast<size_t>(k / 4) * pack_nr
                                             + col0 + n)
                                            * 4
                                    + k % 4];
                        };
                        std::vector<int32_t> shift(
                                static_cast<size_t>(mr) * batch);
                        for (int m = 0; m < mr; ++m)
                            for (int g = 0; g < batch; ++g) {
                                int32_t s = 0;
                                for (int k = 0; k < k_quads * 4; ++k)
                                    s += a_at(m, g * k_quads * 4 + k);
                                shift[static_cast<size_t>(m) * batch + g]
                                        = -128 * s;
                            }
                        const int sa_ld = per_group_src ? batch : 1;
                        const int sa_step = per_group_src ? 1 : 0;
                        std::vector<float> sa(
                                static_cast<size_t>(mr) * batch, 0.0f);
                        for (auto &x : sa)
                            x = scale(rng) * (zero_row ? -1.0f : 1.0f);
                        const int sb_step = nr + 16;
                        const size_t sb_n
                                = static_cast<size_t>(batch) * sb_step;
                        buffer_t sb
                                = nt::make_buffer(sb_n * nt::dt_bytes(sb_dt));
                        for (size_t i = 0; i < sb_n; ++i)
                            nt::store_f32(sb.get(), sb_dt, i, scale(rng));
                        const int ldc = nr + 16, dst_ldc = nr + 32;
                        std::vector<float> c(static_cast<size_t>(mr) * ldc);
                        for (auto &x : c)
                            x = scale(rng) * 100.0f - 0.5f;
                        const std::vector<float> c_in = c;
                        std::vector<uint8_t> dst(
                                static_cast<size_t>(mr) * dst_ldc * 4, 0x5A);
                        const std::vector<uint8_t> dst_in = dst;

                        nat::native_int8_group_ukernel_args_t args;
                        args.A = reinterpret_cast<const uint8_t *>(a.data());
                        args.lda = lda;
                        args.a_step = static_cast<size_t>(k_quads) * 4;
                        args.B = b.data() + col0 * 4;
                        args.b_stride = pack_nr * 4;
                        args.b_step
                                = static_cast<size_t>(k_quads) * pack_nr * 4;
                        args.k_quads = k_quads;
                        args.batch = batch;
                        args.shift = src_u8 ? nullptr : shift.data();
                        args.shift_ld = batch;
                        args.shift_step = 1;
                        args.src_zp = zp;
                        args.src_scale = sa.data();
                        args.src_scale_ld = sa_ld;
                        args.src_scale_step = sa_step;
                        args.wei_scale = sb.get();
                        args.wei_scale_step = sb_step;
                        args.wei_scale_dt = sb_dt;
                        args.C = c.data();
                        args.ldc = ldc;
                        args.accumulate = accumulate;
                        if (store != store_t::c) {
                            args.dst = dst.data();
                            args.dst_ldc = dst_ldc;
                            args.dst_bf16 = store == store_t::bf16;
                        }
                        nat::select_native_int8_group_ukernel(
                                mr, nv, src_u8, wide)(args);

                        size_t bad = 0;
                        for (int m = 0; m < mr; ++m)
                            for (int n = 0; n < nr; ++n) {
                                float sum = accumulate
                                        ? c_in[static_cast<size_t>(m) * ldc + n]
                                        : 0.0f;
                                for (int g = 0; g < batch; ++g) {
                                    int32_t acc = 0;
                                    int64_t exact = 0;
                                    for (int k = 0; k < k_quads * 4; ++k) {
                                        const int kk = g * k_quads * 4 + k;
                                        const int x = a_at(m, kk);
                                        acc += (w_at(kk, n) + 128) * x;
                                        exact += static_cast<int64_t>(
                                                         w_at(kk, n))
                                                * (x - zp);
                                    }
                                    const int64_t d = src_u8 ? exact
                                                             : acc
                                                    + shift[static_cast<size_t>(
                                                                    m)
                                                                    * batch
                                                            + g];
                                    if (d != exact) ++bad;
                                    float t = static_cast<float>(d);
                                    t = t
                                            * sa[static_cast<size_t>(m) * sa_ld
                                                    + g * sa_step];
                                    t = t
                                            * nt::load_f32(sb.get(), sb_dt,
                                                    static_cast<size_t>(g)
                                                                    * sb_step
                                                            + n);
                                    sum = sum + t;
                                }
                                const size_t ci
                                        = static_cast<size_t>(m) * ldc + n;
                                const size_t di
                                        = static_cast<size_t>(m) * dst_ldc + n;
                                if (store == store_t::c) {
                                    if (std::memcmp(&c[ci], &sum, 4) != 0)
                                        ++bad;
                                } else if (store == store_t::f32) {
                                    if (std::memcmp(
                                                dst.data() + 4 * di, &sum, 4)
                                            != 0)
                                        ++bad;
                                } else {
                                    uint16_t want = 0;
                                    nt::store_f32(&want, kBf16, 0, sum);
                                    if (std::memcmp(
                                                dst.data() + 2 * di, &want, 2)
                                            != 0)
                                        ++bad;
                                }
                            }
                        EXPECT_EQ(bad, 0u);
                        // The bytes the tile does not own: C past nr, or all
                        // of C under a destination store; the destination
                        // past nr, or all of it without one.
                        const size_t esz = store == store_t::bf16 ? 2 : 4;
                        size_t written = 0;
                        for (int m = 0; m < mr; ++m)
                            for (int n = 0; n < ldc; ++n) {
                                const size_t ci
                                        = static_cast<size_t>(m) * ldc + n;
                                if (store == store_t::c && n < nr) continue;
                                written += std::memcmp(&c[ci], &c_in[ci], 4)
                                        != 0;
                            }
                        for (size_t i = 0; i < dst.size(); ++i) {
                            const size_t row = i / (dst_ldc * esz);
                            const size_t col = i % (dst_ldc * esz) / esz;
                            if (store != store_t::c
                                    && row < static_cast<size_t>(mr)
                                    && col < static_cast<size_t>(nr))
                                continue;
                            written += dst[i] != dst_in[i];
                        }
                        EXPECT_EQ(written, 0u);
                    }
}

} // namespace
