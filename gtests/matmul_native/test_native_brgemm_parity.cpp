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

// Feature parity of the native BRGEMM (ALGO 11) over native-layout weights
// with the single-expert custom-kernel route.  Every case runs the call
// under ALGO 11 and checks that the native BRGEMM served it, then compares the
// result with the reference algorithm and, when the custom kernel serves the
// same call on the default route, with that result (INT8 bit-identical or
// within one u8 step; BF16 within the BF16 band, bit identity reported).
// Gated activations are also compared with the one-op group_matmul_direct
// fused activation.  One instantiation per feature row.

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using gated_t = mm::matmul_gated_act_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kNone = data_type_t::none;

/// Weight / source configuration of a call.
enum class q_t { bf16, dyn_s8, dyn_u8, st_u8, st_s8 };
/// How the weight reaches the call.
struct po_t {
    post_op_type_t type;
    float alpha = NAN; ///< NaN: the post-op default
    float beta = 0.0f;
    data_type_t buf_dt = kF32; ///< binary buffer type
    bool full = true; ///< binary buffer [M, N], else [1, N]
};

struct pcase_t {
    std::string name;
    q_t q = q_t::bf16;
    int M = 1, N = 64, K = 64;
    data_type_t dst = kBf16;
    int threads = 0; ///< 0: every thread
    float alpha = 1.0f, beta = 0.0f;
    bool transA = false, transB = false;
    int pad = 0; ///< extra elements on lda, ldb and ldc
    data_type_t bias = kNone;
    std::vector<po_t> po;
    gated_t gated = gated_t::none;
    int wc = 1; ///< per-call weight cache mode for a raw weight
    int calls = 1; ///< native calls on the same weight
    bool per_call = false; ///< lowoha_algo = native_brgemm, no process pin
    bool per_token = false; ///< static INT8: source scale / zp per token
    bool wei_per_channel = true;
    data_type_t scale_dt = kF32;
};

void PrintTo(const pcase_t &c, std::ostream *os) {
    *os << c.name;
}

bool is_int8(q_t q) {
    return q != q_t::bf16;
}
bool is_dyn(q_t q) {
    return q == q_t::dyn_s8 || q == q_t::dyn_u8;
}

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kF32: return "f32";
        case kBf16: return "bf16";
        case kF16: return "f16";
        case kS8: return "s8";
        case kU8: return "u8";
        default: return "none";
    }
}

const char *q_name(q_t q) {
    switch (q) {
        case q_t::bf16: return "bf16";
        case q_t::dyn_s8: return "dyns8";
        case q_t::dyn_u8: return "dynu8";
        case q_t::st_u8: return "stu8";
        default: return "sts8";
    }
}

const char *po_name(post_op_type_t t) {
    switch (t) {
        case post_op_type_t::elu: return "elu";
        case post_op_type_t::relu: return "relu";
        case post_op_type_t::leaky_relu: return "lrelu";
        case post_op_type_t::gelu_tanh: return "gelutanh";
        case post_op_type_t::gelu_erf: return "geluerf";
        case post_op_type_t::sigmoid: return "sigmoid";
        case post_op_type_t::swish: return "swish";
        case post_op_type_t::tanh: return "tanh";
        case post_op_type_t::square: return "square";
        case post_op_type_t::abs: return "abs";
        case post_op_type_t::sqrt: return "sqrt";
        case post_op_type_t::exp: return "exp";
        case post_op_type_t::log: return "log";
        case post_op_type_t::clip: return "clip";
        case post_op_type_t::binary_add: return "add";
        case post_op_type_t::binary_mul: return "mul";
        case post_op_type_t::mish: return "mish";
        case post_op_type_t::softmax: return "softmax";
        default: return "none";
    }
}

bool is_binary(post_op_type_t t) {
    return t == post_op_type_t::binary_add || t == post_op_type_t::binary_mul;
}

/// Unique gtest-safe name of a case from its fields.
std::string describe(const pcase_t &c) {
    std::string s = std::string(q_name(c.q)) + "_m" + std::to_string(c.M) + "_n"
            + std::to_string(c.N) + "_k" + std::to_string(c.K) + "_"
            + dt_name(c.dst);
    if (c.threads) s += "_t" + std::to_string(c.threads);
    if (c.alpha != 1.0f || c.beta != 0.0f)
        s += "_a" + std::to_string(int(c.alpha * 10)) + "b"
                + std::to_string(int(c.beta * 10));
    if (c.transA) s += "_ta";
    if (c.transB) s += "_tb";
    if (c.pad) s += "_pad" + std::to_string(c.pad);
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    for (const po_t &p : c.po) {
        s += std::string("_") + po_name(p.type);
        if (!std::isnan(p.alpha)) s += std::to_string(int(p.alpha * 100));
        if (is_binary(p.type))
            s += std::string(dt_name(p.buf_dt)) + (p.full ? "full" : "row");
    }
    if (c.gated == gated_t::silu_and_mul) s += "_silu";
    if (c.gated == gated_t::gelu_and_mul) s += "_gelu";
    if (c.gated == gated_t::swiglu_oai_mul) s += "_swiglu";
    if (c.wc != 1 || c.calls != 1)
        s += "_wc" + std::to_string(c.wc) + "x" + std::to_string(c.calls);
    if (c.per_call) s += "_percall";
    if (c.per_token) s += "_tok";
    if (is_int8(c.q) && !c.wei_per_channel) s += "_wscalar";
    if (is_int8(c.q) && c.scale_dt != kF32)
        s += std::string("_s") + dt_name(c.scale_dt);
    return s;
}

std::vector<pcase_t> named(std::vector<pcase_t> v) {
    std::vector<std::string> seen;
    for (pcase_t &c : v) {
        std::string n = describe(c);
        for (char &ch : n)
            if (ch == '-') ch = 'm';
        int dup = 0;
        for (const std::string &s : seen)
            dup += s == n || s.rfind(n + "_dup", 0) == 0;
        if (dup) n += "_dup" + std::to_string(dup);
        seen.push_back(n);
        c.name = n;
    }
    return v;
}

uint32_t seed_of(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char ch : s)
        h = (h ^ static_cast<uint8_t>(ch)) * 16777619u;
    return h;
}

struct operands_t {
    data_type_t src_dt = kBf16, wei_dt = kBf16;
    int lda = 0, ldb = 0, ldc = 0;
    buffer_t src, wei, bias, wei_scale, src_scale, src_zp, dst_scale, dst_zp;
    std::vector<buffer_t> po_bufs;
    mm::matmul_params params;
};

void set_quant(mm::matmul_quantization_params_t::matmul_quant_t &q,
        const buffer_t &b, data_type_t dt, std::vector<int64_t> dims) {
    q.buff = b.get();
    q.dt = dt;
    q.dims = std::move(dims);
}

operands_t make_operands(const pcase_t &c) {
    operands_t o;
    const uint32_t seed = seed_of(c.name);
    const bool int8 = is_int8(c.q);
    o.src_dt = c.q == q_t::st_u8 ? kU8 : (c.q == q_t::st_s8 ? kS8 : kBf16);
    o.wei_dt = int8 ? kS8 : kBf16;
    const int a_rows = c.transA ? c.K : c.M, a_cols = c.transA ? c.M : c.K;
    const int b_rows = c.transB ? c.N : c.K, b_cols = c.transB ? c.K : c.N;
    o.lda = a_cols + c.pad;
    o.ldb = b_cols + c.pad;
    o.ldc = (c.gated == gated_t::none ? c.N : c.N / 2) + c.pad;
    const size_t a_n = static_cast<size_t>(a_rows) * o.lda;
    if (o.src_dt == kU8)
        o.src = nt::random_buffer(kU8, a_n, seed, 96, 160);
    else if (o.src_dt == kS8)
        o.src = nt::random_buffer(kS8, a_n, seed, -64, 64);
    else
        o.src = nt::random_buffer(kBf16, a_n, seed, -1.0f, 1.0f);
    const size_t b_n = static_cast<size_t>(b_rows) * o.ldb;
    o.wei = int8 ? nt::random_buffer(kS8, b_n, seed + 1, -127, 127)
                 : nt::random_buffer(kBf16, b_n, seed + 1, -1.0f, 1.0f);
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, c.N, seed + 2, -1.0f, 1.0f);

    mm::matmul_params &p = o.params;
    p.dtypes.src = o.src_dt;
    p.dtypes.wei = o.wei_dt;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    auto &qp = p.quant_params;
    if (int8) {
        const int ws = c.wei_per_channel ? c.N : 1;
        o.wei_scale
                = nt::random_buffer(c.scale_dt, ws, seed + 3, 0.002f, 0.01f);
        set_quant(qp.wei_scale, o.wei_scale, c.scale_dt, {1, ws});
        if (is_dyn(c.q)) {
            p.dynamic_quant = true;
            p.dtypes.compute = c.q == q_t::dyn_s8 ? kS8 : kU8;
            // matmul_direct takes per-token source scales only with
            // per-channel weight scales.
            const int rows = c.wei_per_channel ? c.M : 1;
            o.src_scale = nt::make_buffer(static_cast<size_t>(rows) * 4);
            set_quant(qp.src_scale, o.src_scale, kF32, {rows, 1});
            if (c.q == q_t::dyn_u8) {
                o.src_zp = nt::make_buffer(static_cast<size_t>(rows) * 4);
                set_quant(qp.src_zp, o.src_zp, kS32, {rows, 1});
            }
        } else {
            const int ss = c.per_token ? c.M : 1;
            o.src_scale = nt::random_buffer(
                    c.scale_dt, ss, seed + 4, 0.005f, 0.02f);
            set_quant(qp.src_scale, o.src_scale, c.scale_dt, {ss, 1});
            if (c.q == q_t::st_u8) {
                o.src_zp = nt::random_buffer(kS32, ss, seed + 5, 120, 136);
                set_quant(qp.src_zp, o.src_zp, kS32, {ss, 1});
            }
        }
    }
    if (c.dst == kU8 || c.dst == kS8) {
        o.dst_scale = nt::make_buffer(4);
        nt::store_f32(o.dst_scale.get(), kF32, 0, 4.0f);
        set_quant(qp.dst_scale, o.dst_scale, kF32, {1, 1});
        o.dst_zp = nt::make_buffer(4);
        nt::store_f32(o.dst_zp.get(), kS32, 0, c.dst == kU8 ? 128.0f : 0.0f);
        set_quant(qp.dst_zp, o.dst_zp, kS32, {1, 1});
    }
    for (size_t i = 0; i < c.po.size(); ++i) {
        mm::matmul_post_op m;
        m.po_type = c.po[i].type;
        m.alpha = c.po[i].alpha;
        m.beta = c.po[i].beta;
        if (is_binary(c.po[i].type)) {
            const int rows = c.po[i].full ? c.M : 1;
            const int cols = c.gated == gated_t::none ? c.N : c.N / 2;
            o.po_bufs.push_back(nt::random_buffer(c.po[i].buf_dt,
                    static_cast<size_t>(rows) * cols, seed + 10 + i, -1.0f,
                    1.0f));
            m.buff = o.po_bufs.back().get();
            m.dtype = c.po[i].buf_dt;
            m.dims = {rows, cols};
        }
        p.postop_.push_back(m);
    }
    return o;
}

nt::call_t make_call(const pcase_t &c, const operands_t &o) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K;
    call.transA = c.transA;
    call.transB = c.transB;
    call.lda = o.lda;
    call.ldb = o.ldb;
    call.ldc = o.ldc;
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = o.wei.get();
    call.bias = c.bias == kNone ? nullptr : o.bias.get();
    call.params = o.params;
    call.gated = c.gated;
    return call;
}

buffer_t copy_of(const buffer_t &b) {
    buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

/// True when matmul_direct takes the call on this CPU: an f16 bias needs
/// AVX512-FP16.  The case lists hold only calls it takes on some CPU: no
/// per-token source scale of an asymmetric source (an asymmetric dynamic
/// quantization of more than one row included), of a destination other than
/// f32 / bf16 or with per-tensor weight scales, and no mish on a dynamically
/// quantized source.
bool direct_accepts(const pcase_t &c) {
    return c.bias != kF16 || zendnnl_platform_info().get_avx512_f16_status();
}

/// True when the single-expert custom-kernel route admits the call and
/// matmul_direct accepts it on the default route.
bool ck_admits(const pcase_t &c) {
    if (c.alpha != 1.0f || c.beta != 0.0f || c.transA) return false;
    if (c.gated != gated_t::none || c.q == q_t::st_s8) return false;
    if (is_int8(c.q) && c.K % 4 != 0) return false;
    if (c.q == q_t::st_u8 && c.per_token) return false;
    // matmul_direct refuses an f16 bias without AVX512-FP16, and a per-token
    // source scale (dynamic quantization of more than one row) unless the
    // source is symmetric and the weight scales are per channel.
    if (c.bias == kF16 && !zendnnl_platform_info().get_avx512_f16_status())
        return false;
    if (is_dyn(c.q) && c.M > 1 && (c.q == q_t::dyn_u8 || !c.wei_per_channel))
        return false;
    if (c.q != q_t::st_u8) return c.po.empty();
    for (const po_t &p : c.po) {
        const bool eltwise = p.type == post_op_type_t::relu
                || p.type == post_op_type_t::sigmoid;
        const bool binary_full_f32
                = is_binary(p.type) && p.full && p.buf_dt == kF32;
        if (!eltwise && !binary_full_f32) return false;
    }
    return true;
}

mm::grp_matmul_gated_act_t to_group(gated_t g) {
    return static_cast<mm::grp_matmul_gated_act_t>(static_cast<int>(g));
}

class NativeBrgemmParity : public ::testing::TestWithParam<pcase_t> {};

TEST_P(NativeBrgemmParity, Matches) {
    const pcase_t &c = GetParam();
    if (!direct_accepts(c)) GTEST_SKIP() << "matmul_direct refuses the call";
    ::reset_grp_matmul_caches();
    WeightCacheGuard global_cache(2);
    const operands_t o = make_operands(c);
    const bool int8 = is_int8(c.q);
    const int cols = c.gated == gated_t::none ? c.N : c.N / 2;
    // N spare elements: a route that ignores gated_act writes N columns per row
    const size_t out_n = static_cast<size_t>(c.M) * o.ldc + c.N;
    const buffer_t init
            = nt::random_buffer(c.dst, out_n, seed_of(c.name) + 7, -1.0f, 1.0f);
    const buffer_t raw = copy_of(o.wei);
    buffer_t dst = copy_of(init);

    nt::call_t call = make_call(c, o);
    call.dst = dst.get();
    call.params.weight_cache_type = c.wc;
    if (c.per_call) call.params.lowoha_algo = matmul_algo_t::native_brgemm;
    // reference algorithm (gated: pre-activation, then the activation here)
    buffer_t ref = copy_of(init);
    nt::call_t rc = make_call(c, o);
    if (c.gated == gated_t::none) {
        ASSERT_EQ(
                nt::run_reference(rc, raw.get(), ref.get()), status_t::success);
    } else {
        buffer_t pre = nt::make_buffer(static_cast<size_t>(c.M) * c.N * 4);
        ASSERT_EQ(
                nt::run_reference(rc, raw.get(), pre.get()), status_t::success);
        std::vector<float> act(static_cast<size_t>(c.M) * cols);
        nt::gated_act_ref(pre.as<float>(), c.M, c.N, c.gated, act.data());
        for (int m = 0; m < c.M; ++m)
            for (int i = 0; i < cols; ++i)
                nt::store_f32(ref.get(), c.dst,
                        static_cast<size_t>(m) * o.ldc + i,
                        act[static_cast<size_t>(m) * cols + i]);
    }
    const bool int_dst = c.dst == kU8 || c.dst == kS8;
    const float abs_bound = nt::comparison_abs_bound(
            c.K, epsilon_bf16, c.alpha, c.beta, true, int8, false);

    {
        std::unique_ptr<nt::algo_pin_t> pin;
        if (!c.per_call)
            pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
        for (int i = 0; i < c.calls; ++i) {
            std::memcpy(dst.get(), init.get(), init.bytes);
            ASSERT_EQ(nt::run_matmul(call), status_t::success)
                    << "ALGO 11 call " << i;
            if (c.wc != 0) {
                ASSERT_GE(nt::native_layouts(o.wei.get()), 1u)
                        << "ALGO 11 call " << i
                        << " was not served by the native BRGEMM over the "
                           "native VNNI layout";
            }
            if (int_dst) {
                EXPECT_TRUE(nt::within_steps(dst.get(), o.ldc, ref.get(), o.ldc,
                        c.dst, c.M, cols, 1))
                        << "ALGO 11 call " << i << " vs reference";
            } else {
                EXPECT_TRUE(nt::within_band(dst.get(), o.ldc, ref.get(), o.ldc,
                        c.dst, c.M, cols, rtol_bf16, abs_bound))
                        << "ALGO 11 call " << i << " vs reference";
            }
        }
    }

    // The custom-kernel route and the one-op group call run on the default
    // route, which a process-wide algorithm replaces for the whole process.
    if ((ck_admits(c) || c.gated != gated_t::none)
            && !nt::default_route_reachable()) {
        RecordProperty("ck", "not_served (process-wide algorithm)");
        ::reset_grp_matmul_caches();
        return;
    }

    // the single-expert custom-kernel route on the same call
    if (ck_admits(c)) {
        buffer_t ck_dst = copy_of(init);
        buffer_t ck_wei = copy_of(raw);
        nt::call_t kc = make_call(c, o);
        kc.wei = ck_wei.get();
        kc.dst = ck_dst.get();
        kc.params.weight_cache_type = 2;
        ::reset_grp_matmul_caches();
        nt::ck_route_pin_t route;
        nt::ck_engagement_t engagement;
        ASSERT_EQ(nt::run_matmul(kc), status_t::success) << "default route";
        if (!engagement.engaged()) {
            RecordProperty("ck",
                    "not_served (the default route did not take the custom "
                    "kernel)");
        } else if (int8) {
            EXPECT_TRUE(nt::within_steps(dst.get(), o.ldc, ck_dst.get(), o.ldc,
                    c.dst, c.M, cols, int_dst ? 1 : 0))
                    << "vs the custom-kernel route";
            RecordProperty("ck", "served");
        } else {
            EXPECT_TRUE(nt::within_band(dst.get(), o.ldc, ck_dst.get(), o.ldc,
                    c.dst, c.M, cols, rtol_bf16, abs_bound))
                    << "vs the custom-kernel route";
            const ::testing::AssertionResult bits = nt::within_steps(
                    dst.get(), o.ldc, ck_dst.get(), o.ldc, c.dst, c.M, cols, 0);
            RecordProperty("ck",
                    std::string("served bits=")
                            + (bits ? "identical" : bits.message()));
        }
    }

    // gated: the one-op group call with the fused activation
    if (c.gated != gated_t::none) {
        buffer_t g_dst = nt::make_buffer(
                static_cast<size_t>(c.M) * c.N * nt::dt_bytes(c.dst));
        buffer_t g_wei = nt::kept_until_exit(copy_of(raw));
        nt::call_t gc = make_call(c, o);
        gc.wei = g_wei.get();
        gc.gated = gated_t::none;
        gc.params.weight_cache_type = 2;
        ::reset_grp_matmul_caches();
        nt::ck_route_pin_t route;
        nt::ck_engagement_t engagement;
        ASSERT_EQ(nt::run_group_one_op(gc, to_group(c.gated), g_dst.get()),
                status_t::success)
                << "one-op group_matmul_direct";
        EXPECT_TRUE(nt::within_band(dst.get(), o.ldc, g_dst.get(), c.N, c.dst,
                c.M, cols, rtol_bf16, abs_bound))
                << "vs one-op group_matmul_direct (custom kernel "
                << (engagement.engaged() ? "served" : "not served") << ")";
    }
    ::reset_grp_matmul_caches();
}

pcase_t mk(q_t q, int M, int N, int K, data_type_t dst = kBf16) {
    pcase_t c;
    c.q = q;
    c.M = M;
    c.N = N;
    c.K = K;
    c.dst = dst;
    return c;
}

std::vector<pcase_t> bf16_cases() {
    std::vector<pcase_t> v;
    for (int M : {1, 2, 4, 7, 8, 16, 32, 63, 64, 65, 128, 512, 1500})
        v.push_back(mk(q_t::bf16, M, 4096, 1000));
    for (int M : {1, 65, 1500})
        v.push_back(mk(q_t::bf16, M, 4096, 1000, kF32));
    for (int K : {1, 3, 31, 63, 64, 100, 1000, 4096})
        for (int M : {4, 65})
            v.push_back(mk(q_t::bf16, M, 96, K, M == 4 ? kBf16 : kF32));
    for (int N : {32, 64, 96, 4096})
        for (int M : {8, 128})
            v.push_back(mk(q_t::bf16, M, N, 64));
    v.push_back(mk(q_t::bf16, 1, 11008, 4096));
    v.push_back(mk(q_t::bf16, 32, 11008, 4096));
    v.push_back(mk(q_t::bf16, 128, 4096, 11008 / 2 * 2));
    for (int t : {1, 4}) {
        pcase_t a = mk(q_t::bf16, 16, 4096, 1000);
        a.threads = t;
        v.push_back(a);
        pcase_t b = mk(q_t::bf16, 512, 96, 63);
        b.threads = t;
        v.push_back(b);
        pcase_t d = mk(q_t::bf16, 1, 11008, 4096, kF32);
        d.threads = t;
        v.push_back(d);
    }
    return named(v);
}

/// An asymmetric (u8) dynamic quantization runs on one row, the only row
/// count matmul_direct takes it with per-token scales.
std::vector<pcase_t> int8_dyn_cases() {
    std::vector<pcase_t> v;
    for (q_t q : {q_t::dyn_s8, q_t::dyn_u8}) {
        const bool rows = q == q_t::dyn_s8;
        for (int M : {1, 4, 16, 64, 65, 512})
            if (rows || M == 1) v.push_back(mk(q, M, 4096, 1024));
        for (int K : {1, 3, 4, 31, 63, 64, 100, 1000, 4096})
            v.push_back(mk(q, rows ? 8 : 1, 96, K, kF32));
        for (int N : {32, 64})
            v.push_back(mk(q, rows ? 4 : 1, N, 64));
        v.push_back(mk(q, rows ? 4 : 1, 11008, 4096));
        if (rows) v.push_back(mk(q, 1500, 96, 100, kF32));
        pcase_t s = mk(q, rows ? 8 : 1, 4096, 1024);
        s.scale_dt = kBf16;
        v.push_back(s);
        pcase_t w = mk(q, rows ? 8 : 1, 4096, 1024);
        w.wei_per_channel = false;
        v.push_back(w);
        for (int t : {1, 4}) {
            pcase_t a = mk(q, rows ? 16 : 1, 4096, 1024);
            a.threads = t;
            v.push_back(a);
        }
    }
    return named(v);
}

std::vector<pcase_t> int8_static_cases() {
    std::vector<pcase_t> v;
    for (data_type_t d : {kBf16, kF32, kU8})
        for (int M : {1, 16, 128})
            v.push_back(mk(q_t::st_u8, M, 4096, 1024, d));
    for (data_type_t d : {kBf16, kF32, kS8, kU8})
        v.push_back(mk(q_t::st_s8, 8, 4096, 1024, d));
    // Per-token source scales: a symmetric source and an f32 / bf16 dst.
    for (data_type_t d : {kBf16, kF32}) {
        pcase_t a = mk(q_t::st_s8, 65, 96, 100, d);
        a.per_token = true;
        v.push_back(a);
    }
    for (bool pc : {false, true}) {
        pcase_t a = mk(q_t::st_u8, 8, 4096, 1024);
        a.wei_per_channel = pc;
        a.scale_dt = kBf16;
        v.push_back(a);
    }
    for (int K : {3, 31, 100, 1000})
        v.push_back(mk(q_t::st_u8, 8, 96, K, kU8));
    v.push_back(mk(q_t::st_u8, 1500, 96, 100, kU8));
    return named(v);
}

std::vector<pcase_t> alpha_beta_cases() {
    std::vector<pcase_t> v;
    const float ab[][2] = {{0.0f, 0.0f}, {0.5f, 0.0f}, {2.0f, 0.0f},
            {1.0f, 1.0f}, {1.0f, 0.5f}, {0.5f, 2.0f}, {0.0f, 1.0f}};
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (const auto &x : ab) {
            pcase_t c = mk(q, 16, 96, 100, q == q_t::bf16 ? kBf16 : kF32);
            c.alpha = x[0];
            c.beta = x[1];
            v.push_back(c);
        }
    return named(v);
}

std::vector<pcase_t> trans_a_cases() {
    std::vector<pcase_t> v;
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (int M : {7, 65})
            for (int pad : {0, 8}) {
                pcase_t c = mk(q, M, 96, 100, kF32);
                c.transA = true;
                c.pad = pad;
                v.push_back(c);
            }
    return named(v);
}

std::vector<pcase_t> ld_cases() {
    std::vector<pcase_t> v;
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (int M : {8, 65})
            for (int pad : {1, 16}) {
                pcase_t c = mk(q, M, 96, 100);
                c.pad = pad;
                v.push_back(c);
            }
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (int pad : {0, 5}) {
            pcase_t c = mk(q, 8, 4096, 1024);
            c.transB = true;
            c.pad = pad;
            v.push_back(c);
        }
    return named(v);
}

std::vector<pcase_t> bias_cases() {
    std::vector<pcase_t> v;
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (data_type_t b : {kBf16, kF32, kF16})
            for (int M : {4, 128}) {
                pcase_t c = mk(q, M, 4096, 1024);
                c.bias = b;
                v.push_back(c);
            }
    return named(v);
}

std::vector<pcase_t> postop_cases() {
    std::vector<pcase_t> v;
    const std::vector<std::vector<po_t>> eltwise = {
            {{post_op_type_t::relu}},
            {{post_op_type_t::leaky_relu, 0.1f}},
            {{post_op_type_t::elu}},
            {{post_op_type_t::elu, 0.5f}},
            {{post_op_type_t::gelu_tanh}},
            {{post_op_type_t::gelu_erf}},
            {{post_op_type_t::sigmoid}},
            {{post_op_type_t::swish}},
            {{post_op_type_t::swish, 0.5f}},
            {{post_op_type_t::swish, 1.702f}},
            {{post_op_type_t::tanh}},
            {{post_op_type_t::square}},
            {{post_op_type_t::abs}, {post_op_type_t::sqrt}},
            {{post_op_type_t::tanh}, {post_op_type_t::exp}},
            {{post_op_type_t::exp}, {post_op_type_t::log}},
            {{post_op_type_t::clip, -0.5f, 0.5f}},
            {{post_op_type_t::mish}},
            {{post_op_type_t::softmax}},
            {{post_op_type_t::gelu_erf}, {post_op_type_t::softmax}},
    };
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (const auto &chain : eltwise) {
            // matmul_direct takes no mish on a dynamically quantized source.
            if (is_dyn(q) && chain[0].type == post_op_type_t::mish) continue;
            pcase_t c = mk(q, 16, 128, 64,
                    q == q_t::bf16 ? kBf16 : (q == q_t::dyn_s8 ? kF32 : kU8));
            c.po = chain;
            v.push_back(c);
        }
    for (q_t q : {q_t::bf16, q_t::st_u8})
        for (post_op_type_t t :
                {post_op_type_t::binary_add, post_op_type_t::binary_mul})
            for (data_type_t bd : {kF32, kBf16})
                for (bool full : {true, false}) {
                    pcase_t c
                            = mk(q, 16, 128, 64, q == q_t::bf16 ? kBf16 : kF32);
                    po_t p {t};
                    p.buf_dt = bd;
                    p.full = full;
                    c.po = {p};
                    v.push_back(c);
                }
    po_t mul_full {post_op_type_t::binary_mul};
    po_t add_full {post_op_type_t::binary_add};
    po_t add_row {post_op_type_t::binary_add};
    add_row.full = false;
    const std::vector<std::vector<po_t>> chains = {
            {{post_op_type_t::relu}, add_full},
            {{post_op_type_t::gelu_erf}, mul_full,
                    {post_op_type_t::clip, -0.5f, 0.5f}},
            {{post_op_type_t::swish, 0.5f}, {post_op_type_t::tanh}},
            {mul_full, add_full},
            {add_row, {post_op_type_t::sigmoid}},
    };
    for (q_t q : {q_t::bf16, q_t::dyn_u8, q_t::st_u8})
        for (const auto &chain : chains) {
            pcase_t c = mk(q, q == q_t::dyn_u8 ? 1 : 8, 4096, 1024,
                    q == q_t::st_u8 ? kU8 : kBf16);
            c.po = chain;
            v.push_back(c);
        }
    return named(v);
}

std::vector<pcase_t> gated_cases() {
    std::vector<pcase_t> v;
    const int shapes[][3] = {{1, 4096, 1000}, {8, 64, 64}, {65, 4096, 64}};
    int i = 0;
    for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul})
        for (q_t q : {q_t::bf16, q_t::dyn_s8})
            for (data_type_t b : {kNone, kF32})
                for (data_type_t d : {kBf16, kF32}) {
                    const int *s = shapes[i++ % 3];
                    pcase_t c = mk(q, s[0], s[1], s[2], d);
                    c.gated = g;
                    c.bias = b;
                    v.push_back(c);
                }
    return named(v);
}

std::vector<pcase_t> weight_cache_cases() {
    std::vector<pcase_t> v;
    for (int wc : {0, 1, 2})
        for (int K : {1024, 1023}) {
            pcase_t c = mk(q_t::bf16, 4, 4096, K);
            c.wc = wc;
            c.calls = 2;
            v.push_back(c);
        }
    for (int wc : {0, 1, 2})
        for (q_t q : {q_t::dyn_s8, q_t::st_u8}) {
            pcase_t c = mk(q, 4, 4096, 1024);
            c.wc = wc;
            c.calls = 2;
            v.push_back(c);
        }
    pcase_t prompt = mk(q_t::bf16, 512, 4096, 1024);
    prompt.wc = 2;
    prompt.calls = 2;
    v.push_back(prompt);
    return named(v);
}

std::vector<pcase_t> per_call_cases() {
    std::vector<pcase_t> v;
    for (q_t q : {q_t::bf16, q_t::dyn_s8, q_t::st_u8})
        for (int M : {1, 128}) {
            pcase_t c = mk(q, M, 4096, 1024);
            c.per_call = true;
            v.push_back(c);
        }
    return named(v);
}

std::string case_name(const ::testing::TestParamInfo<pcase_t> &info) {
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
        Bf16, NativeBrgemmParity, ::testing::ValuesIn(bf16_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(Int8Dyn, NativeBrgemmParity,
        ::testing::ValuesIn(int8_dyn_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(Int8Static, NativeBrgemmParity,
        ::testing::ValuesIn(int8_static_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(AlphaBeta, NativeBrgemmParity,
        ::testing::ValuesIn(alpha_beta_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(TransA, NativeBrgemmParity,
        ::testing::ValuesIn(trans_a_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(
        Ld, NativeBrgemmParity, ::testing::ValuesIn(ld_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(
        Bias, NativeBrgemmParity, ::testing::ValuesIn(bias_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(PostOp, NativeBrgemmParity,
        ::testing::ValuesIn(postop_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(Gated, NativeBrgemmParity,
        ::testing::ValuesIn(gated_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(WeightCache, NativeBrgemmParity,
        ::testing::ValuesIn(weight_cache_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(PerCallAlgo, NativeBrgemmParity,
        ::testing::ValuesIn(per_call_cases()), case_name);

} // namespace
