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
// (ALGO 11): matmul_direct calls with a per-group {M, G}, per-token or
// per-tensor source scale over a static s8 source or a bf16 / f32 source
// quantized in the call.  Every case checks that the native BRGEMM served
// the call, then compares it with
//   * the reference algorithm's order modelled here (the exact integer sum
//     of each group, then f32(sum) * src scale * wei scale added in group
//     order), bit for bit;
//   * the reference algorithm itself for a per-group source, bit for bit;
//   * an f64 model within its rounding bound for a per-token or per-tensor
//     source (the reference algorithm sums such a call over all of K);
//   * the default route within the sym-quant band, when the process is not
//     pinned to an algorithm.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
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

/// Static s8, or bf16 / f32 quantized to s8 in the call.
enum class src_t { s8, bf16, f32 };
/// Source scale: {1, 1}, {M, 1} or {M, G}.
enum class gran_t { tensor, token, group };
/// Post-ops: binary add of an [M, N] f32 matrix, gelu (erf), or both.
enum class po_t { none, add, gelu, gelu_add };

struct gcase_t {
    std::string name;
    int M = 16, N = 64, gs = 32, G = 4; ///< K = gs * G
    src_t src = src_t::s8;
    gran_t gran = gran_t::group;
    bool transA = false, transB = false;
    data_type_t dst = kF32, bias = kNone;
    data_type_t wei_scale_dt = kF32, src_scale_dt = kF32;
    po_t po = po_t::none;
    float alpha = 1.0f, beta = 0.0f;
    int wc = 1; ///< weight cache mode of the call
    int threads = 4; ///< 0: every thread
    int calls = 1; ///< native calls on the same weight
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
        default: return "none";
    }
}

std::string describe(const gcase_t &c) {
    static const char *const kSrc[] = {"s8", "dynbf16", "dynf32"};
    static const char *const kGran[] = {"tensor", "token", "group"};
    static const char *const kPo[] = {"", "_add", "_gelu", "_geluadd"};
    std::string s = std::string(kSrc[static_cast<int>(c.src)]) + "_"
            + kGran[static_cast<int>(c.gran)] + "_m" + std::to_string(c.M)
            + "_n" + std::to_string(c.N) + "_gs" + std::to_string(c.gs) + "x"
            + std::to_string(c.G) + "_" + dt_name(c.dst);
    if (c.transA) s += "_ta";
    if (c.transB) s += "_tb";
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    s += kPo[static_cast<int>(c.po)];
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
    buffer_t src, wei, bias, wei_scale, src_scale, bin, init;
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
            : c.src == src_t::bf16                ? kBf16
                                                  : kF32;
    o.src = src_dt == kS8 ? nt::random_buffer(kS8, a_n, seed, -128, 127)
                          : nt::random_buffer(src_dt, a_n, seed, -1.0f, 1.0f);
    o.wei = nt::random_buffer(
            kS8, static_cast<size_t>(K) * N, seed + 1, -128, 127);
    o.wei_scale = nt::random_buffer(c.wei_scale_dt,
            static_cast<size_t>(c.G) * N, seed + 2, 0.002f, 0.01f);
    mm::matmul_params &p = o.params;
    p.dtypes.src = src_dt;
    p.dtypes.wei = kS8;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    auto &qp = p.quant_params;
    set_quant(qp.wei_scale, o.wei_scale.get(), c.wei_scale_dt, {c.G, N});
    const int64_t n_scale = src_scale_count(c);
    if (c.src == src_t::s8) {
        o.src_scale = nt::random_buffer(
                c.src_scale_dt, n_scale, seed + 3, 0.002f, 0.01f);
    } else {
        p.dynamic_quant = true;
        p.dtypes.compute = kS8;
        o.src_scale = nt::make_buffer(n_scale * nt::dt_bytes(c.src_scale_dt));
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
    o.init = nt::random_buffer(
            c.dst, static_cast<size_t>(M) * N, seed + 6, -2.0f, 2.0f);
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

/// The s8 source rows [M][K] and their scales [M][G] the native route
/// multiplies: a static source as given, a dynamic one quantized here by
/// the shared quantizer from its rows made row-major, as the route does.
struct source_t {
    std::vector<int8_t> a;
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
    buffer_t written;
    if (c.src == src_t::s8) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                s.a[static_cast<size_t>(m) * K + k]
                        = o.src.as<int8_t>()[at(m, k)];
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
        const void *src = rows.get();
        int quantized_lda = K;
        size_t elem_size = es;
        mm::matmul_batch_params_t batch;
        mm::reorder_quant_buffers_t buffers;
        EXPECT_EQ(mm::reorder_quantization_wrapper(src, K, quantized_lda,
                          elem_size, q, batch, false, M, K, 1, buffers),
                status_t::success);
        EXPECT_EQ(q.dtypes.src, kS8);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                s.a[static_cast<size_t>(m) * K + k]
                        = static_cast<const int8_t *>(
                                src)[static_cast<size_t>(m) * quantized_lda
                                + k];
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
        int32_t d = 0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
            d += static_cast<int32_t>(s.a[static_cast<size_t>(m) * K + k])
                    * weight_at(c, o, k, n);
        float v = static_cast<float>(d);
        v = v * s.scale[static_cast<size_t>(m) * G + g];
        v = v
                * nt::load_f32(o.wei_scale.get(), c.wei_scale_dt,
                        static_cast<size_t>(g) * c.N + n);
        sum = sum + v;
    }
    const size_t mn = static_cast<size_t>(m) * c.N + n;
    if (c.alpha != 1.0f) sum *= c.alpha;
    if (c.beta != 0.0f) sum += nt::load_f32(o.init.get(), c.dst, mn) * c.beta;
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
            d += static_cast<int64_t>(s.a[static_cast<size_t>(m) * K + k])
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

/// The values the call stores, in dst order.
buffer_t model_dst(const gcase_t &c, const operands_t &o, const source_t &s) {
    buffer_t out = nt::make_buffer(
            static_cast<size_t>(c.M) * c.N * nt::dt_bytes(c.dst));
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n)
            nt::store_f32(out.get(), c.dst, static_cast<size_t>(m) * c.N + n,
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
    return nat::make_native_problem(
            desc, c.params, c.bias, nat::kNativeLayoutPackNr);
}

/// The call under ALGO 11 into a copy of the initial dst, calls times; the
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
    if (c.gran != gran_t::group && !transcendental(c) && c.po != po_t::add) {
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
    // group sizes (12 and 96 not powers of two) over the source scales
    for (int gs : {4, 8, 12, 32, 64, 96, 128, 256})
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
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt8PerGroup, NativeBrgemmInt8PerGroup,
        ::testing::ValuesIn(cases()),
        [](const ::testing::TestParamInfo<gcase_t> &info) {
            return info.param.name;
        });

// A gated activation over a pack of interleaved gate / up columns: the
// group sums and scales of both halves in the reference order, then the
// activation within the band of the epilogue suite.
TEST(NativeBrgemmInt8PerGroupGated, MatchesTheModel) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul})
        for (data_type_t dst : {kF32, kBf16}) {
            gcase_t c;
            c.M = 7;
            c.N = 128;
            c.dst = dst;
            c.bias = kBf16;
            c.wei_scale_dt = kBf16;
            c.name = std::string("gated")
                    + std::to_string(static_cast<int>(act)) + "_"
                    + dt_name(dst);
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            const source_t s = source_of(c, o);
            const int I = c.N / 2;
            buffer_t got = nt::make_buffer(
                    static_cast<size_t>(c.M) * I * nt::dt_bytes(dst));
            const buffer_t scale = copy_of(o.src_scale);
            nt::call_t call = make_call(c, o, o.wei.get(), got.get(), scale);
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
            EXPECT_TRUE(nt::within_band(got.get(), I, want.get(), I, dst, c.M,
                    I, dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f, 3e-4f));
        }
    mm::clear_matmul_weight_caches();
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
            {"direct_store=0", false}, {"threads_cap=1", true},
            {"schedule=row_groups,domain_threads=2", true},
            {"int8_kc=64,int8_mr=5", true}};
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
    p.quant_params.src_scale.dims = {1, 1};
    EXPECT_EQ(check(p), status_t::unimplemented) << "a u8 source";
    p = call(kS8, 4);
    int32_t zp = 0;
    set_quant(p.quant_params.src_zp, &zp, kS32, {1, 1});
    EXPECT_EQ(check(p), status_t::unimplemented) << "a source zero point";
    for (data_type_t dst : {kU8, kS8, kS32, kF16}) {
        p = call(kS8, 4);
        p.dtypes.dst = dst;
        EXPECT_EQ(check(p), status_t::unimplemented) << dt_name(dst);
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
    EXPECT_EQ(check(p), status_t::unimplemented) << "bf16 quantized to u8";

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

// Calls the native BRGEMM does not serve: the call takes the route it takes
// without the native BRGEMM.
TEST(NativeBrgemmInt8PerGroupDecline, OtherRoutesServe) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    WeightCacheGuard global_cache(2);
    struct decline_t {
        const char *name;
        gcase_t c;
        data_type_t src_dt; ///< kNone: the case's
        bool src_zp;
    };
    std::vector<decline_t> v;
    {
        gcase_t c;
        c.gs = 6;
        c.G = 8;
        v.push_back({"group_size_6", c, kNone, false});
        c.gs = 10;
        c.G = 4;
        v.push_back({"group_size_10", c, kNone, false});
    }
    {
        gcase_t c;
        c.gran = gran_t::tensor;
        v.push_back({"u8_source", c, kU8, true});
        v.push_back({"s8_source_zero_point", c, kNone, true});
    }
    for (decline_t &d : v) {
        SCOPED_TRACE(d.name);
        d.c.name = d.name;
        operands_t o = make_operands(d.c);
        int32_t zp = 128;
        if (d.src_dt == kU8) {
            o.src = nt::random_buffer(
                    kU8, static_cast<size_t>(d.c.M) * d.c.K(), 7, 0, 255);
            o.params.dtypes.src = kU8;
        }
        if (d.src_zp)
            set_quant(o.params.quant_params.src_zp, &zp, kS32, {1, 1});
        buffer_t dst = copy_of(o.init);
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        EXPECT_EQ(nt::run_matmul(make_call(
                          d.c, o, o.wei.get(), dst.get(), o.src_scale)),
                status_t::success);
        EXPECT_EQ(nt::native_layouts(o.wei.get()), 0u)
                << "served by the native BRGEMM";
    }
    // matmul_direct refuses a per-group source with per-channel weight
    // scales on every route; the tile turns it away too.
    {
        gcase_t c;
        c.name = "group_source_channel_weight";
        operands_t o = make_operands(c);
        set_quant(o.params.quant_params.wei_scale, o.wei_scale.get(), kF32,
                {1, c.N});
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
                status_t::unimplemented);
    }
    mm::clear_matmul_weight_caches();
}

} // namespace
