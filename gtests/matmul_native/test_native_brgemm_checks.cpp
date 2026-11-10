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

// What matmul_direct's checks take only for a call that asks for the native
// BRGEMM (lowoha_algo native_brgemm, or ZENDNNL_MATMUL_ALGO=10): such a call
// runs on the native BRGEMM with the result of the same call on its own;
// with the algorithm unset the checks refuse it, and so do the default
// route's checks of a call the native BRGEMM hands back under
// ZENDNNL_MATMUL_ALGO=10.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kS4 = data_type_t::s4;

bool native_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512f && u.avx512bf16 && u.avx512vnni;
}

/// The quantized calls these tests draw: an s8 weight with a bf16 source
/// the call quantizes to s8 per token or to u8 per tensor (with its zero
/// point), or with a static s8 source per token or per group; an s4 weight
/// with a bf16 source the call quantizes to s8 per token or per group, or
/// with a static s8 source per token or per group.
enum class form_t {
    dyn_s8_token,
    dyn_u8_tensor,
    st_s8_token,
    st_s8_group,
    s4_dyn_token,
    s4_dyn_group,
    s4_st_token,
    s4_st_group
};

const char *form_name(form_t f) {
    switch (f) {
        case form_t::dyn_s8_token: return "s8 weight, bf16 source to s8";
        case form_t::dyn_u8_tensor: return "s8 weight, bf16 source to u8";
        case form_t::st_s8_token: return "s8 weight, s8 source per token";
        case form_t::st_s8_group: return "s8 weight, s8 source per group";
        case form_t::s4_dyn_token: return "s4 weight, bf16 source per token";
        case form_t::s4_dyn_group: return "s4 weight, bf16 source per group";
        case form_t::s4_st_token: return "s4 weight, s8 source per token";
        case form_t::s4_st_group: return "s4 weight, s8 source per group";
    }
    return "";
}

const form_t kForms[] = {form_t::dyn_s8_token, form_t::dyn_u8_tensor,
        form_t::st_s8_token, form_t::st_s8_group, form_t::s4_dyn_token,
        form_t::s4_dyn_group, form_t::s4_st_token, form_t::s4_st_group};

bool s4_form(form_t f) {
    return f == form_t::s4_dyn_token || f == form_t::s4_dyn_group
            || f == form_t::s4_st_token || f == form_t::s4_st_group;
}

bool dynamic_form(form_t f) {
    return f == form_t::dyn_s8_token || f == form_t::dyn_u8_tensor
            || f == form_t::s4_dyn_token || f == form_t::s4_dyn_group;
}

bool group_form(form_t f) {
    return f == form_t::st_s8_group || f == form_t::s4_dyn_group
            || f == form_t::s4_st_group;
}

/// One call of a form, M x N x K with G groups over K for its per-group
/// scales, into a dst of dst_dt, on its own operands; the weight scales of
/// an s8 weight are per column unless the source's are per group or
/// wei_groups.
struct qcall_t {
    nt::call_t c;
    buffer_t src, wei, dst, prior, src_scale, wei_scale, src_zp;
    int G = 1;
};

std::unique_ptr<qcall_t> make_qcall(form_t f, int M, int N, int K, int G,
        data_type_t dst_dt, uint32_t seed, bool wei_groups = false) {
    auto q = std::make_unique<qcall_t>();
    q->G = G;
    const bool s4 = s4_form(f);
    const bool dynamic = dynamic_form(f);
    const size_t a_n = static_cast<size_t>(M) * K;
    q->src = dynamic ? nt::random_buffer(kBf16, a_n, seed, -1.0f, 1.0f)
                     : nt::random_buffer(kS8, a_n, seed, -127.0f, 127.0f);
    q->wei = nt::kept_until_exit(s4
                    ? nt::random_buffer(kU8, static_cast<size_t>(N) * K / 2,
                              seed + 1, 0, 255)
                    : nt::random_buffer(kS8, static_cast<size_t>(K) * N,
                              seed + 1, -127.0f, 127.0f));
    const bool per_group_wei = s4 || wei_groups || group_form(f);
    const int64_t n_wei_scale = per_group_wei ? static_cast<int64_t>(G) * N : N;
    q->wei_scale
            = nt::random_buffer(kF32, n_wei_scale, seed + 2, 0.002f, 0.01f);
    const bool src_groups = group_form(f);
    const int64_t n_src_scale = f == form_t::dyn_u8_tensor ? 1
            : src_groups                                   ? int64_t {M} * G
                                                           : M;
    q->src_scale = dynamic
            ? nt::make_buffer(n_src_scale * sizeof(float))
            : nt::random_buffer(kF32, n_src_scale, seed + 3, 0.002f, 0.01f);
    const size_t c_n = static_cast<size_t>(M) * N;
    q->prior = nt::random_buffer(
            dst_dt, c_n, seed + 4, dst_dt == kU8 ? 0.0f : -2.0f, 2.0f);
    q->dst = nt::make_buffer(q->prior.bytes);
    std::memcpy(q->dst.get(), q->prior.get(), q->prior.bytes);
    nt::call_t &c = q->c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.transB = s4;
    c.src = q->src.get();
    c.wei = q->wei.get();
    c.dst = q->dst.get();
    c.params.dtypes.src = dynamic ? kBf16 : kS8;
    c.params.dtypes.wei = s4 ? kS4 : kS8;
    c.params.dtypes.dst = dst_dt;
    c.params.weight_cache_type = 1;
    if (dynamic || s4) {
        c.params.dynamic_quant = dynamic;
        c.params.dtypes.compute = f == form_t::dyn_u8_tensor ? kU8 : kS8;
    }
    auto &qp = c.params.quant_params;
    qp.wei_scale.buff = q->wei_scale.get();
    qp.wei_scale.dt = kF32;
    qp.wei_scale.dims = per_group_wei ? std::vector<int64_t> {G, N}
                                      : std::vector<int64_t> {1, N};
    qp.src_scale.buff = q->src_scale.get();
    qp.src_scale.dt = kF32;
    qp.src_scale.dims = f == form_t::dyn_u8_tensor ? std::vector<int64_t> {1, 1}
            : src_groups                           ? std::vector<int64_t> {M, G}
                         : std::vector<int64_t> {M, 1};
    if (f == form_t::dyn_u8_tensor) {
        q->src_zp = nt::make_buffer(sizeof(int32_t));
        qp.src_zp.buff = q->src_zp.get();
        qp.src_zp.dt = kS32;
        qp.src_zp.dims = {1, 1};
    }
    return q;
}

/// c with its dst at dst and a source scale buffer of its own, which a
/// dynamic quantization writes.
nt::call_t with_dst(const qcall_t &q, void *dst, buffer_t &scale) {
    nt::call_t c = q.c;
    scale = nt::make_buffer(q.src_scale.bytes);
    std::memcpy(scale.get(), q.src_scale.get(), q.src_scale.bytes);
    c.params.quant_params.src_scale.buff = scale.get();
    c.dst = dst;
    return c;
}

/// c as a gated call (silu_and_mul) with beta over its dst of N / 2
/// columns, which the native BRGEMM hands back under ZENDNNL_MATMUL_ALGO=10.
nt::call_t gated_with_beta(nt::call_t c) {
    c.gated = mm::matmul_gated_act_t::silu_and_mul;
    c.beta = 0.5f;
    return c;
}

/// The call on native_brgemm_execute over its raw weight, packed for the
/// call only: matmul_direct's checks of the call do not run.
status_t run_on_native(const nt::call_t &c) {
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
    nat::native_weight_ref_t ref;
    ref.weight = c.wei;
    ref.pack_nr = nat::kNativeLayoutPackNr;
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(
            desc, ref, c.src, c.dst, c.bias, c.params);
}

/// mish(x) = x * tanh(softplus(x)), in f32.
float mish_of(float x) {
    const float sp = std::fmax(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
    return x * std::tanh(sp);
}

/// The two ways a call asks for the native BRGEMM: with lowoha_algo, or
/// under ZENDNNL_MATMUL_ALGO=10 without it.
struct request_t {
    const char *name;
    bool per_call;
};
const request_t kRequests[] = {
        {"lowoha_algo native_brgemm", true}, {"ZENDNNL_MATMUL_ALGO=10", false}};

/// c asking for the native BRGEMM as r says, under the process-wide
/// algorithm of r (pin_of).
nt::call_t asking(const nt::call_t &c, const request_t &r) {
    nt::call_t a = c;
    a.params.lowoha_algo
            = r.per_call ? matmul_algo_t::native_brgemm : matmul_algo_t::none;
    return a;
}

matmul_algo_t pin_of(const request_t &r) {
    return r.per_call ? matmul_algo_t::none : matmul_algo_t::native_brgemm;
}

// mish on the quantized calls of the native BRGEMM: matmul_direct takes it
// for a call that asks for the native BRGEMM, which applies it to the
// result of the same call without it (an f32 dst, within the band of the
// epilogue's mish); with the algorithm unset, on a GGML pack that names the
// native BRGEMM, and on a gated call with beta over a dst of N / 2 columns
// the native BRGEMM hands back under ZENDNNL_MATMUL_ALGO=10, the checks
// refuse it, dst untouched.
TEST(NativeBrgemmChecks, MishOnQuantizedCalls) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const int M = 6, N = 64, K = 128, G = 4;
    uint32_t seed = 9100;
    for (const form_t f : kForms) {
        auto q = make_qcall(f, M, N, K, G, kF32, seed += 8);
        mm::matmul_post_op mish;
        mish.po_type = post_op_type_t::mish;
        for (const request_t &r : kRequests) {
            SCOPED_TRACE(std::string(form_name(f)) + ", " + r.name);
            const nt::algo_pin_t pin(pin_of(r));
            buffer_t pre = nt::make_buffer(q->prior.bytes);
            buffer_t got = nt::make_buffer(q->prior.bytes);
            buffer_t s0, s1;
            nt::call_t plain = asking(with_dst(*q, pre.get(), s0), r);
            ASSERT_EQ(nt::run_matmul(plain), status_t::success);
            nt::call_t c = asking(with_dst(*q, got.get(), s1), r);
            c.params.postop_.push_back(mish);
            ASSERT_EQ(nt::run_matmul(c), status_t::success);
            int bad = 0;
            for (int i = 0; i < M * N; ++i) {
                const float want = mish_of(pre.as<float>()[i]);
                const float g = got.as<float>()[i];
                if (!(std::fabs(g - want) <= 1e-5f + 1e-4f * std::fabs(want))
                        && bad++ < 4)
                    ADD_FAILURE()
                            << "element " << i << ": " << g << " for " << want;
            }
            EXPECT_EQ(bad, 0);
        }
        SCOPED_TRACE(form_name(f));
        buffer_t s;
        nt::call_t c = with_dst(*q, q->dst.get(), s);
        c.params.postop_.push_back(mish);
        {
            const nt::algo_pin_t unset(matmul_algo_t::none);
            EXPECT_EQ(nt::run_matmul(c), status_t::failure);
            nt::call_t ggml = c;
            ggml.params.packing.pack_format_b = 1;
            ggml.params.lowoha_algo = matmul_algo_t::native_brgemm;
            EXPECT_EQ(nt::run_matmul(ggml), status_t::failure);
        }
        {
            const nt::algo_pin_t algo10(matmul_algo_t::native_brgemm);
            EXPECT_EQ(nt::run_matmul(gated_with_beta(c)), status_t::failure);
        }
        EXPECT_EQ(std::memcmp(q->dst.get(), q->prior.get(), q->prior.bytes), 0)
                << "a refused call wrote dst";
    }
}

// f16 source scales of W4A8: matmul_direct takes them for a call that asks
// for the native BRGEMM, which serves it with the bits of the same call on
// native_brgemm_execute (a bf16 source quantized in the call per token or
// per group, or a static s8 source); a static source has the bits of the
// same call with f32 scales of the same values.  With the algorithm unset,
// and on a gated call with beta over a dst of N / 2 columns the native
// BRGEMM hands back under ZENDNNL_MATMUL_ALGO=10, the checks refuse them,
// dst untouched.
TEST(NativeBrgemmChecks, F16SourceScalesOfW4A8) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const int M = 5, N = 64, K = 256, G = 4;
    uint32_t seed = 9300;
    for (const form_t f : {form_t::s4_dyn_token, form_t::s4_dyn_group,
                 form_t::s4_st_token, form_t::s4_st_group})
        for (const data_type_t dst : {kBf16, kF32}) {
            auto q = make_qcall(f, M, N, K, G, dst, seed += 8);
            const int64_t n_scale = q->src_scale.bytes / sizeof(float);
            buffer_t f16_scale = dynamic_form(f)
                    ? nt::make_buffer(n_scale * 2)
                    : nt::random_buffer(kF16, n_scale, seed + 5, 0.002f, 0.01f);
            buffer_t f32_scale = nt::make_buffer(n_scale * sizeof(float));
            for (int64_t i = 0; i < n_scale; ++i)
                f32_scale.as<float>()[i]
                        = nt::load_f32(f16_scale.get(), kF16, i);
            q->src_scale = f16_scale;
            q->c.params.quant_params.src_scale.buff = f16_scale.get();
            q->c.params.quant_params.src_scale.dt = kF16;
            for (const request_t &r : kRequests) {
                SCOPED_TRACE(std::string(form_name(f)) + " into "
                        + (dst == kBf16 ? "bf16" : "f32") + ", " + r.name);
                const nt::algo_pin_t pin(pin_of(r));
                buffer_t got = nt::make_buffer(q->prior.bytes);
                buffer_t want = nt::make_buffer(q->prior.bytes);
                buffer_t s0, s1;
                ASSERT_EQ(
                        nt::run_matmul(asking(with_dst(*q, got.get(), s0), r)),
                        status_t::success);
                ASSERT_EQ(run_on_native(with_dst(*q, want.get(), s1)),
                        status_t::success);
                EXPECT_EQ(std::memcmp(got.get(), want.get(), got.bytes), 0)
                        << "differs from the call on the native BRGEMM";
                if (dynamic_form(f)) continue;
                buffer_t as_f32 = nt::make_buffer(q->prior.bytes);
                nt::call_t c = asking(q->c, r);
                c.dst = as_f32.get();
                c.params.quant_params.src_scale.buff = f32_scale.get();
                c.params.quant_params.src_scale.dt = kF32;
                ASSERT_EQ(nt::run_matmul(c), status_t::success);
                EXPECT_EQ(std::memcmp(got.get(), as_f32.get(), got.bytes), 0)
                        << "differs from the call with f32 scales";
            }
            SCOPED_TRACE(form_name(f));
            buffer_t s;
            const nt::call_t c = with_dst(*q, q->dst.get(), s);
            {
                const nt::algo_pin_t unset(matmul_algo_t::none);
                EXPECT_EQ(nt::run_matmul(c), status_t::failure);
            }
            {
                const nt::algo_pin_t algo10(matmul_algo_t::native_brgemm);
                EXPECT_EQ(
                        nt::run_matmul(gated_with_beta(c)), status_t::failure);
            }
            EXPECT_EQ(std::memcmp(q->dst.get(), q->prior.get(), q->prior.bytes),
                    0)
                    << "a refused call wrote dst";
        }
}

// u8 / s8 / s32 destinations of INT8 calls with per-token or per-group
// source scales: matmul_direct takes them for a call that asks for the
// native BRGEMM, which serves it with the bits of the same call on
// native_brgemm_execute, within one step of the same call into f32 stored
// as an integer dst (the dst scale, then the zero point, then the nearest
// integer of its range): a static s8 or a bf16 source quantized in the call
// per token, over per-column or per-group weight scales, and a static s8
// source per group, with and without a dst scale and zero point.  With the
// algorithm unset, and on a gated call with beta over a dst of N / 2 columns
// the native BRGEMM hands back under ZENDNNL_MATMUL_ALGO=10, the checks
// refuse them; an f16 dst stays refused.
TEST(NativeBrgemmChecks, IntegerDestinationsOfPerTokenAndPerGroupCalls) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const int M = 7, N = 64, K = 128, G = 4;
    struct kind_t {
        form_t form;
        bool wei_groups;
    };
    const kind_t kinds[] = {{form_t::st_s8_token, false},
            {form_t::dyn_s8_token, false}, {form_t::st_s8_token, true},
            {form_t::dyn_s8_token, true}, {form_t::st_s8_group, true}};
    uint32_t seed = 9500;
    for (const kind_t &k : kinds)
        for (const data_type_t dst : {kU8, kS8, kS32})
            for (const bool requant : {false, true}) {
                if (requant && dst == kS32) continue;
                auto q = make_qcall(
                        k.form, M, N, K, G, dst, seed += 8, k.wei_groups);
                const float dst_scale = 4.0f;
                const int32_t dst_zp = dst == kU8 ? 128 : (dst == kS8 ? -3 : 0);
                if (requant) {
                    auto &qp = q->c.params.quant_params;
                    qp.dst_scale.buff = &dst_scale;
                    qp.dst_scale.dt = kF32;
                    qp.dst_scale.dims = {1, 1};
                    qp.dst_zp.buff = &dst_zp;
                    qp.dst_zp.dt = kS32;
                    qp.dst_zp.dims = {1, 1};
                }
                const std::string name = std::string(form_name(k.form))
                        + (k.wei_groups ? ", weight groups" : "") + " into "
                        + (dst == kU8                ? "u8"
                                        : dst == kS8 ? "s8"
                                                     : "s32")
                        + (requant ? " requantized" : "");
                for (const request_t &r : kRequests) {
                    SCOPED_TRACE(name + ", " + r.name);
                    const nt::algo_pin_t pin(pin_of(r));
                    buffer_t got = nt::make_buffer(q->prior.bytes);
                    buffer_t want = nt::make_buffer(q->prior.bytes);
                    buffer_t as_f32 = nt::make_buffer(
                            static_cast<size_t>(M) * N * sizeof(float));
                    buffer_t s0, s1, s2;
                    ASSERT_EQ(nt::run_matmul(
                                      asking(with_dst(*q, got.get(), s0), r)),
                            status_t::success);
                    ASSERT_EQ(run_on_native(with_dst(*q, want.get(), s1)),
                            status_t::success);
                    EXPECT_EQ(std::memcmp(got.get(), want.get(), got.bytes), 0)
                            << "differs from the call on the native BRGEMM";
                    nt::call_t f = asking(with_dst(*q, as_f32.get(), s2), r);
                    f.params.dtypes.dst = kF32;
                    f.params.quant_params.dst_scale = {};
                    f.params.quant_params.dst_zp = {};
                    ASSERT_EQ(nt::run_matmul(f), status_t::success);
                    const float lo = dst == kU8 ? 0.0f
                            : dst == kS8        ? -128.0f
                                                : -2147483648.0f;
                    const float hi = dst == kU8 ? 255.0f
                            : dst == kS8        ? 127.0f
                                                : 2147483520.0f;
                    int bad = 0;
                    for (int i = 0; i < M * N; ++i) {
                        float v = as_f32.as<float>()[i];
                        if (requant)
                            v = v * dst_scale + static_cast<float>(dst_zp);
                        const float model = std::nearbyint(
                                std::fmin(std::fmax(v, lo), hi));
                        const float g = nt::load_f32(got.get(), dst, i);
                        if (!(std::fabs(g - model) <= 1.0f) && bad++ < 4)
                            ADD_FAILURE() << "element " << i << ": " << g
                                          << " for " << model;
                    }
                    EXPECT_EQ(bad, 0);
                }
                SCOPED_TRACE(name);
                buffer_t s;
                const nt::call_t c = with_dst(*q, q->dst.get(), s);
                {
                    const nt::algo_pin_t unset(matmul_algo_t::none);
                    EXPECT_EQ(nt::run_matmul(c), status_t::failure);
                }
                {
                    // Without AOCL-DLP, weight groups over a per-token
                    // source keep the native error: the reference kernel
                    // does not compute them.
                    const nt::algo_pin_t algo10(matmul_algo_t::native_brgemm);
                    EXPECT_EQ(nt::run_matmul(gated_with_beta(c)),
                            !ZENDNNL_DEPENDS_AOCLDLP && k.wei_groups
                                            && k.form != form_t::st_s8_group
                                    ? status_t::unimplemented
                                    : status_t::failure);
                }
                EXPECT_EQ(std::memcmp(
                                  q->dst.get(), q->prior.get(), q->prior.bytes),
                        0)
                        << "a refused call wrote dst";
            }
    // An f16 dst of the same calls stays refused (isa_unsupported without
    // AVX512-FP16).
    for (const kind_t &k : kinds)
        for (const request_t &r : kRequests) {
            SCOPED_TRACE(std::string(form_name(k.form))
                    + (k.wei_groups ? ", weight groups" : "") + " into f16, "
                    + r.name);
            auto q = make_qcall(
                    k.form, M, N, K, G, kBf16, seed += 8, k.wei_groups);
            const nt::algo_pin_t pin(pin_of(r));
            buffer_t s;
            nt::call_t c = asking(with_dst(*q, q->dst.get(), s), r);
            c.params.dtypes.dst = data_type_t::f16;
            EXPECT_NE(nt::run_matmul(c), status_t::success);
            EXPECT_EQ(std::memcmp(q->dst.get(), q->prior.get(), q->prior.bytes),
                    0)
                    << "a refused call wrote dst";
        }
}

} // namespace
