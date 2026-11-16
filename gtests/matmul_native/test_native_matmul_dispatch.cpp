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

// The routes of calls whose kernel is native_brgemm: the algorithm values
// that select it, and the entry and the kernels a call takes.

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/auto_tuner/auto_tuner.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_expert_slice.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

mm::matmul_params dtype_params(data_type_t src, data_type_t wei) {
    mm::matmul_params p;
    p.dtypes.src = src;
    p.dtypes.wei = wei;
    p.dtypes.dst = data_type_t::f32;
    return p;
}

bool gemv_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512bf16 && u.avx512vnni && u.avx512f;
}

/// One dense call of a route: operands, params and a second output for the
/// result of the kernel it should run on.  The weight stays allocated until
/// the process exits, so that no later weight takes its address in the
/// pointer-keyed layout cache.
struct route_call_t {
    nt::call_t c;
    nt::buffer_t src, wei, dst, ref;
    float src_scale = 0.5f, wei_scale = 0.25f;
    std::vector<float> wei_scales;

    route_call_t(data_type_t src_dt, data_type_t wei_dt, data_type_t dst_dt,
            int M, int N, int K, int threads, uint32_t seed,
            bool transB = false) {
        const bool int8 = wei_dt == data_type_t::s8;
        const bool s8_src = src_dt == data_type_t::s8;
        src = nt::random_buffer(src_dt, static_cast<size_t>(M) * K, seed,
                int8 ? (s8_src ? -8.0f : 0.0f) : -1.0f, int8 ? 8.0f : 1.0f);
        wei = nt::kept_until_exit(
                nt::random_buffer(wei_dt, static_cast<size_t>(K) * N, seed + 1,
                        int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f));
        dst = nt::make_buffer(
                static_cast<size_t>(M) * N * nt::dt_bytes(dst_dt));
        ref = nt::make_buffer(
                static_cast<size_t>(M) * N * nt::dt_bytes(dst_dt));
        c.M = M;
        c.N = N;
        c.K = K;
        c.transB = transB;
        c.src = src.get();
        c.wei = wei.get();
        c.dst = dst.get();
        c.is_weights_const = true;
        c.params = dtype_params(src_dt, wei_dt);
        c.params.dtypes.dst = dst_dt;
        c.params.num_threads = threads;
        if (int8) {
            auto &q = c.params.quant_params;
            q.src_scale.buff = &src_scale;
            q.src_scale.dt = data_type_t::f32;
            q.src_scale.dims = {1, 1};
            q.wei_scale.buff = &wei_scale;
            q.wei_scale.dt = data_type_t::f32;
            q.wei_scale.dims = {1, 1};
        }
    }

    /// Weight scales of n elements, {1, n}.
    void wei_scale_cols(int n) {
        wei_scales.assign(n, 0.25f);
        c.params.quant_params.wei_scale.buff = wei_scales.data();
        c.params.quant_params.wei_scale.dims = {1, n};
    }

    /// dst and ref hold identical bits.
    ::testing::AssertionResult same_bits() const {
        return nt::within_steps(dst.get(), c.N, ref.get(), c.N,
                c.params.dtypes.dst, c.M, c.N, 0);
    }

    /// out against the reference algorithm over the call's weight, within
    /// the band of its data types; out and ref start from prior (which beta
    /// reads), and the reference result goes to ref.
    ::testing::AssertionResult near_reference(
            const void *out, const nt::buffer_t &prior) const {
        std::memcpy(ref.get(), prior.get(), ref.bytes);
        if (nt::run_reference(c, c.wei, ref.get()) != status_t::success)
            return ::testing::AssertionFailure() << "no reference result";
        const data_type_t dt = c.params.dtypes.dst;
        const bool int8 = c.params.dtypes.wei == data_type_t::s8;
        return nt::within_band(out, c.N, ref.get(), c.N, dt, c.M, c.N,
                rtol_bf16,
                nt::comparison_abs_bound(
                        c.K, epsilon_bf16, 1.0f, c.beta, true, int8, false));
    }
};

/// The call of r on the native BRGEMM (matmul_direct), from the dst it
/// holds: a layout of its weight, within the band of the reference.
void expect_served_on_a_layout(route_call_t &r) {
    // The layout check needs a cached layout whatever the process's mode.
    const WeightCacheGuard cache(2);
    const nt::buffer_t prior = nt::make_buffer(r.dst.bytes);
    std::memcpy(prior.get(), r.dst.get(), r.dst.bytes);
    ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
    EXPECT_GE(nt::native_layouts(r.c.wei), 1u);
    EXPECT_TRUE(r.near_reference(r.dst.get(), prior));
}

// A call that asks for the native BRGEMM at M = 1 with N <= 4 on a [K, N]
// weight reads the native layout at weight cache mode 1, per call and under
// ZENDNNL_MATMUL_ALGO=10: a layout of its weight, within the band of the
// reference.
TEST(NativeMatmulDispatch, RequestedNarrowCallsReadTheNativeLayout) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32;
    struct case_t {
        int N, K, threads;
        data_type_t dst;
        bool bias, relu;
        float beta;
    };
    const case_t cases[] = {
            {1, 234, 1, f32, false, false, 0.0f},
            {2, 64, 8, bf16, true, false, 0.0f},
            {3, 6000, 1, f32, true, true, 0.0f},
            {4, 77, 32, bf16, false, true, 0.5f},
    };
    uint32_t seed = 6500;
    for (const bool per_call : {true, false})
        for (const case_t &cs : cases) {
            SCOPED_TRACE(std::string(per_call ? "lowoha_algo native_brgemm"
                                              : "ZENDNNL_MATMUL_ALGO=10")
                    + " N " + std::to_string(cs.N) + " K "
                    + std::to_string(cs.K));
            std::unique_ptr<nt::algo_pin_t> pin;
            if (!per_call)
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            route_call_t r(
                    bf16, bf16, cs.dst, 1, cs.N, cs.K, cs.threads, seed += 4);
            r.c.params.weight_cache_type = 1;
            const nt::buffer_t bias
                    = nt::random_buffer(f32, cs.N, seed + 2, -1.0f, 1.0f);
            if (cs.bias) {
                r.c.bias = bias.get();
                r.c.params.dtypes.bias = f32;
            }
            if (cs.relu) {
                mm::matmul_post_op relu;
                relu.po_type = post_op_type_t::relu;
                relu.alpha = 0.0f;
                r.c.params.postop_ = {relu};
            }
            r.c.beta = cs.beta;
            if (cs.beta != 0.0f) {
                const nt::buffer_t prior = nt::random_buffer(
                        cs.dst, cs.N, seed + 3, -1.0f, 1.0f);
                std::memcpy(r.dst.get(), prior.get(), r.dst.bytes);
            }
            if (per_call) r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            expect_served_on_a_layout(r);
        }
}

// A call that asks for the native BRGEMM at M = 1, bf16 or INT8, of 64 to
// 128 columns on one thread or eight, reads the native layout at weight
// cache mode 1, per call and under ZENDNNL_MATMUL_ALGO=10: a layout of its
// weight, within the band of the reference.
TEST(NativeMatmulDispatch, RequestedOneRowCallsReadTheNativeLayout) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32,
                      u8 = data_type_t::u8, s8 = data_type_t::s8;
    struct case_t {
        data_type_t src, dst;
        int N, K, threads;
        bool bias, relu;
        const char *name;
    };
    const case_t cases[] = {
            {bf16, bf16, 96, 300, 1, false, false, "bf16"},
            {bf16, f32, 64, 512, 8, true, true, "bf16"},
            {u8, f32, 128, 512, 1, true, false, "u8 x s8"},
            {s8, bf16, 64, 256, 8, false, true, "s8 x s8"},
    };
    uint32_t seed = 6700;
    for (const bool per_call : {true, false})
        for (const case_t &cs : cases) {
            SCOPED_TRACE(std::string(per_call ? "lowoha_algo native_brgemm"
                                              : "ZENDNNL_MATMUL_ALGO=10")
                    + " " + cs.name + " N " + std::to_string(cs.N) + " K "
                    + std::to_string(cs.K) + " threads "
                    + std::to_string(cs.threads));
            std::unique_ptr<nt::algo_pin_t> pin;
            if (!per_call)
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            const bool int8 = cs.src != bf16;
            route_call_t r(cs.src, int8 ? s8 : bf16, cs.dst, 1, cs.N, cs.K,
                    cs.threads, seed += 4);
            r.c.params.weight_cache_type = 1;
            const nt::buffer_t bias
                    = nt::random_buffer(f32, cs.N, seed + 2, -1.0f, 1.0f);
            if (cs.bias) {
                r.c.bias = bias.get();
                r.c.params.dtypes.bias = f32;
            }
            if (cs.relu) {
                mm::matmul_post_op relu;
                relu.po_type = post_op_type_t::relu;
                relu.alpha = 0.0f;
                r.c.params.postop_ = {relu};
            }
            if (per_call) r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            expect_served_on_a_layout(r);
        }
}

// A u8 / s8 call that asks for the native BRGEMM at M = 1 with bf16 or f16
// weight scales (per tensor or of N elements), per call and under
// ZENDNNL_MATMUL_ALGO=10, is served with a layout cached for its weight and
// gives the reference's output within the INT8 band.
TEST(NativeMatmulDispatch, RequestedInt8CallsWithF16OrBf16ScalesAreServed) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const data_type_t s8 = data_type_t::s8, f32 = data_type_t::f32;
    const int N = 96, K = 256;
    uint32_t seed = 6900;
    for (const bool per_call : {true, false})
        for (const data_type_t src : {data_type_t::u8, s8})
            for (const data_type_t scale_dt :
                    {data_type_t::bf16, data_type_t::f16})
                for (const int scales : {1, N}) {
                    SCOPED_TRACE(
                            std::string(per_call ? "lowoha_algo native_brgemm, "
                                                 : "ZENDNNL_MATMUL_ALGO=10, ")
                            + (src == s8 ? "s8" : "u8") + " x s8, "
                            + (scale_dt == data_type_t::bf16 ? "bf16" : "f16")
                            + " weight scales: " + std::to_string(scales));
                    std::unique_ptr<nt::algo_pin_t> pin;
                    if (!per_call)
                        pin.reset(new nt::algo_pin_t(
                                matmul_algo_t::native_brgemm));
                    route_call_t r(src, s8, f32, 1, N, K, 1, seed += 4);
                    if (scales == N) r.wei_scale_cols(N);
                    if (per_call)
                        r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
                    const nt::buffer_t wei_scales = nt::random_buffer(scale_dt,
                            static_cast<size_t>(scales), seed + 2, 0.125f,
                            0.5f);
                    auto &ws = r.c.params.quant_params.wei_scale;
                    ws.buff = wei_scales.get();
                    ws.dt = scale_dt;
                    ws.dims = {1, scales};
                    ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
                    EXPECT_GE(nt::native_layouts(r.c.wei), 1u);
                    ASSERT_EQ(nt::run_reference(r.c, r.c.wei, r.ref.get()),
                            status_t::success);
                    EXPECT_TRUE(nt::within_band(r.dst.get(), N, r.ref.get(), N,
                            f32, 1, N, rtol_f32,
                            nt::comparison_abs_bound(K, epsilon_f32, 1.0f, 0.0f,
                                    false, true, false)));
                }
}

// A u8 / s8 call that asks for the native BRGEMM at M = 1 on a constant
// weight with alpha other than 1 or beta other than 0, per call and under
// ZENDNNL_MATMUL_ALGO=10, is served with a layout cached for its weight and
// the bits of the same call on a weight that is not constant, and gives the
// reference's output within the INT8 band.  The calls carry a bias, source
// and weight scales other than 1, a dst that beta reads, and with a u8
// source a zero point.
TEST(NativeMatmulDispatch, RequestedInt8CallsWithAlphaOrBetaAreServed) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const data_type_t s8 = data_type_t::s8, u8 = data_type_t::u8,
                      f32 = data_type_t::f32, bf16 = data_type_t::bf16;
    const int N = 96, K = 256;
    const struct {
        float alpha, beta;
    } scalars[] = {{2.0f, 0.0f}, {1.0f, 0.5f}, {0.5f, 1.0f}};
    const int32_t zp = 3;
    uint32_t seed = 7000;
    for (const bool per_call : {true, false})
        for (const data_type_t src : {u8, s8})
            for (const data_type_t dst : {f32, bf16})
                for (const auto &ab : scalars) {
                    SCOPED_TRACE(
                            std::string(per_call ? "lowoha_algo native_brgemm, "
                                                 : "ZENDNNL_MATMUL_ALGO=10, ")
                            + (src == s8 ? "s8" : "u8") + " x s8 to "
                            + (dst == f32 ? "f32" : "bf16") + ", alpha "
                            + std::to_string(ab.alpha) + ", beta "
                            + std::to_string(ab.beta));
                    std::unique_ptr<nt::algo_pin_t> pin;
                    if (!per_call)
                        pin.reset(new nt::algo_pin_t(
                                matmul_algo_t::native_brgemm));
                    route_call_t r(src, s8, dst, 1, N, K, 1, seed += 4);
                    if (per_call)
                        r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
                    const nt::buffer_t bias
                            = nt::random_buffer(f32, N, seed + 2, -1.0f, 1.0f);
                    r.c.bias = bias.get();
                    r.c.params.dtypes.bias = f32;
                    if (src == u8) {
                        auto &q = r.c.params.quant_params;
                        q.src_zp.buff = &zp;
                        q.src_zp.dt = data_type_t::s32;
                        q.src_zp.dims = {1, 1};
                    }
                    r.c.alpha = ab.alpha;
                    r.c.beta = ab.beta;
                    // The dst beta reads, the same for every call.
                    const nt::buffer_t prior
                            = nt::random_buffer(dst, N, seed + 3, -1.0f, 1.0f);
                    const nt::buffer_t not_const_dst
                            = nt::make_buffer(r.dst.bytes);
                    std::memcpy(r.dst.get(), prior.get(), r.dst.bytes);
                    std::memcpy(r.ref.get(), prior.get(), r.dst.bytes);
                    std::memcpy(not_const_dst.get(), prior.get(), r.dst.bytes);
                    ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
                    EXPECT_GE(nt::native_layouts(r.c.wei), 1u);
                    nt::call_t not_const = r.c;
                    not_const.is_weights_const = false;
                    not_const.dst = not_const_dst.get();
                    ASSERT_EQ(nt::run_matmul(not_const), status_t::success);
                    EXPECT_TRUE(nt::within_steps(r.dst.get(), N,
                            not_const_dst.get(), N, dst, 1, N, 0))
                            << "against the call on a weight that is not "
                               "constant";
                    ASSERT_EQ(nt::run_reference(r.c, r.c.wei, r.ref.get()),
                            status_t::success);
                    const bool low = dst == bf16;
                    EXPECT_TRUE(nt::within_band(r.dst.get(), N, r.ref.get(), N,
                            dst, 1, N, low ? rtol_bf16 : rtol_f32,
                            nt::comparison_abs_bound(K,
                                    low ? epsilon_bf16 : epsilon_f32, ab.alpha,
                                    ab.beta, low, true, false)));
                }
}

// A call that asks for the native BRGEMM (lowoha_algo native_brgemm) at M = 1
// on a team of eight caches a layout of its weight.
TEST(NativeMatmulDispatch, RequestedCallsCacheTheNativeLayout) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    route_call_t asked(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
            1, 96, 128, 8, 5800);
    asked.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
    asked.c.params.weight_cache_type = 1;
    ASSERT_EQ(nt::run_matmul(asked.c), status_t::success);
    EXPECT_GE(nt::native_layouts(asked.c.wei), 1u);
}

/// Restores the process algo and weight cache mode a test sets.
struct config_restore_t {
    matmul_config_t &cfg = matmul_config_t::instance();
    const int32_t algo = cfg.get_algo();
    const int32_t weight_cache = cfg.get_weight_cache();
    ~config_restore_t() {
        cfg.set_algo(algo);
        cfg.set_weight_cache(weight_cache);
    }
};

// Dynamic dispatch never selects the native BRGEMM, which runs only the
// calls that ask for it: BF16 calls at M 1 .. 8 and INT8 calls at M = 1 on
// one thread take AOCL-DLP blocked, with the bits of the same call on
// lowoha_algo aocl_dlp_blocked and no native layout of the weight.
TEST(NativeMatmulDispatch, DynamicDispatchNeverSelectsTheNativeBrgemm) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    // Dynamic dispatch lowers weight cache mode 2 for the process.
    config_restore_t restore;
    const data_type_t bf16 = data_type_t::bf16, s8 = data_type_t::s8,
                      u8 = data_type_t::u8, f32 = data_type_t::f32;
    struct case_t {
        const char *name;
        data_type_t src, wei, dst;
        int M, N, K, threads;
        bool transB;
    };
    const case_t cases[] = {
            {"bf16 M 1 N 3", bf16, bf16, f32, 1, 3, 200, 1, false},
            {"bf16 M 1 N 100", bf16, bf16, bf16, 1, 100, 128, 1, false},
            {"bf16 M 1 N 256, 8 threads", bf16, bf16, bf16, 1, 256, 128, 8,
                    true},
            {"bf16 M 2", bf16, bf16, f32, 2, 64, 300, 1, false},
            {"bf16 M 8", bf16, bf16, bf16, 8, 96, 64, 4, false},
            {"u8 x s8 M 1", u8, s8, f32, 1, 96, 64, 1, false},
            {"s8 x s8 M 1 to bf16", s8, s8, bf16, 1, 100, 72, 1, true},
    };
    uint32_t seed = 6200;
    for (const case_t &cs : cases) {
        SCOPED_TRACE(cs.name);
        route_call_t r(cs.src, cs.wei, cs.dst, cs.M, cs.N, cs.K, cs.threads,
                seed += 4, cs.transB);
        r.c.params.lowoha_algo = matmul_algo_t::dynamic_dispatch;
        mm::matmul_params p = r.c.params;
        EXPECT_EQ(static_cast<int>(mm::kernel_select(p, 1, 1, 1, cs.M, cs.N,
                          cs.K, cs.threads, nullptr, true)),
                static_cast<int>(matmul_algo_t::aocl_dlp_blocked));
        ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
        EXPECT_EQ(nt::native_layouts(r.c.wei), 0u);
        nt::call_t blocked = r.c;
        blocked.dst = r.ref.get();
        blocked.params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        ASSERT_EQ(nt::run_matmul(blocked), status_t::success);
        EXPECT_TRUE(r.same_bits());
    }
}

// The auto tuner never runs the native BRGEMM: its candidate list skips
// native_brgemm, as it skips dynamic_dispatch, auto_tuner and a value that is
// not a matmul algorithm.
TEST(NativeMatmulDispatch, AutoTunerCandidatesSkipTheNativeBrgemm) {
    nt::expect_in_child(
            {{"ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES", "10,1,0,12,2"}}, [] {
        const std::vector<matmul_algo_t> want = {
                matmul_algo_t::aocl_dlp_blocked, matmul_algo_t::onednn_blocked};
        const std::vector<matmul_algo_t> &got = mm::get_algo_candidates();
        if (got == want) return 0;
        std::fprintf(stderr, "candidates:");
        for (const matmul_algo_t a : got)
            std::fprintf(stderr, " %d", static_cast<int>(a));
        std::fprintf(stderr, "\n");
        return 1;
    }, "ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES");
}

/// The kernel group matmul resolves when the process algorithm selects none
/// of its kernels.
matmul_algo_t group_unset_kernel() {
#if ZENDNNL_DEPENDS_AOCLDLP
    return matmul_algo_t::aocl_dlp_blocked;
#else
    return matmul_algo_t::reference;
#endif
}

// ZENDNNL_MATMUL_ALGO and ZENDNNL_BMM_ALGO take 10 as the native BRGEMM, 11
// as moe_custom_kernel, and 12, which names no algorithm, as any other
// unknown value (13): the config holds algo_count, a call takes the kernel
// and the bits of a call with an unknown per-call lowoha_algo, and the native
// BRGEMM takes no call.  Under 11 a call takes the kernel and the bits of a
// call with lowoha_algo moe_custom_kernel.  Group matmul resolves 10, 12 and
// 13 as unset and 11 as moe_custom_kernel; the auto tuner skips 10, 12 and
// 13 and keeps 11.
TEST(NativeMatmulDispatch,
        Algo10IsTheNativeBrgemm11TheMoeCustomKernelAnd12IsUnknown) {
    EXPECT_EQ(static_cast<int32_t>(matmul_algo_t::native_brgemm), 10);
    EXPECT_EQ(static_cast<int32_t>(matmul_algo_t::moe_custom_kernel), 11);
    EXPECT_TRUE(zendnnl::common::is_valid_matmul_algo(10));
    EXPECT_TRUE(zendnnl::common::is_valid_matmul_algo(11));
    EXPECT_FALSE(zendnnl::common::is_valid_matmul_algo(12));
    EXPECT_FALSE(zendnnl::common::is_valid_matmul_algo(13));
    EXPECT_FALSE(zendnnl::common::is_valid_matmul_algo(-1));
    const bool native_isa = gemv_isa() && nat::native_base_isa();
    for (const char *unknown : {"12", "13"}) {
        const std::string what = std::string("ZENDNNL_MATMUL_ALGO=") + unknown
                + ", ZENDNNL_BMM_ALGO=" + unknown;
        nt::expect_in_child({{"ZENDNNL_MATMUL_ALGO", unknown},
                                    {"ZENDNNL_BMM_ALGO", unknown}},
                [] {
            matmul_config_t &cfg = matmul_config_t::instance();
            const int32_t no_algo
                    = static_cast<int32_t>(matmul_algo_t::algo_count);
            if (cfg.get_algo() != no_algo || cfg.get_bmm_algo() != no_algo
                    || matmul_config_t::is_native_brgemm_algo()) {
                std::fprintf(stderr, "algo %d, bmm algo %d\n", cfg.get_algo(),
                        cfg.get_bmm_algo());
                return 1;
            }
            if (mm::resolve_matmul_backend() != group_unset_kernel()) return 2;
            // A weight that is not constant keeps the call off the one-op
            // group route, whose backend resolves the process algorithm as
            // unset.
            route_call_t r(data_type_t::bf16, data_type_t::bf16,
                    data_type_t::f32, 8, 96, 64, 4, 6600);
            r.c.is_weights_const = false;
            nt::call_t per_call = r.c;
            per_call.dst = r.ref.get();
            per_call.params.lowoha_algo = static_cast<matmul_algo_t>(99);
            for (const int batch : {1, 2}) {
                mm::matmul_params p = r.c.params, q = per_call.params;
                if (mm::kernel_select(p, batch, batch, batch, 8, 96, 64, 4,
                            nullptr, false)
                        != mm::kernel_select(q, batch, batch, batch, 8, 96, 64,
                                4, nullptr, false))
                    return 3;
            }
            if (nt::run_matmul(r.c) != status_t::success
                    || nt::run_matmul(per_call) != status_t::success
                    || !r.same_bits())
                return 4;
            route_call_t kept(data_type_t::bf16, data_type_t::bf16,
                    data_type_t::f32, 8, 96, 64, 4, 6700);
            kept.c.params.weight_cache_type = 1;
            if (nt::run_matmul(kept.c) != status_t::success
                    || nt::native_layouts(kept.c.wei) != 0)
                return 5;
            return 0;
        }, what);
    }
    nt::expect_in_child(
            {{"ZENDNNL_MATMUL_ALGO", "11"}, {"ZENDNNL_BMM_ALGO", "11"}}, [] {
        matmul_config_t &cfg = matmul_config_t::instance();
        const matmul_algo_t custom = matmul_algo_t::moe_custom_kernel;
        if (cfg.get_algo() != static_cast<int32_t>(custom)
                || cfg.get_bmm_algo() != static_cast<int32_t>(custom)
                || matmul_config_t::is_native_brgemm_algo()) {
            std::fprintf(stderr, "algo %d, bmm algo %d\n", cfg.get_algo(),
                    cfg.get_bmm_algo());
            return 1;
        }
        if (mm::resolve_matmul_backend() != custom) return 2;
        route_call_t r(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
                8, 96, 64, 4, 6900);
        r.c.is_weights_const = false;
        nt::call_t per_call = r.c;
        per_call.dst = r.ref.get();
        per_call.params.lowoha_algo = custom;
        for (const int batch : {1, 2}) {
            mm::matmul_params p = r.c.params, q = per_call.params;
            if (mm::kernel_select(
                        p, batch, batch, batch, 8, 96, 64, 4, nullptr, false)
                            != custom
                    || mm::kernel_select(q, batch, batch, batch, 8, 96, 64, 4,
                               nullptr, false)
                            != custom)
                return 3;
        }
        if (nt::run_matmul(r.c) != status_t::success
                || nt::run_matmul(per_call) != status_t::success
                || !r.same_bits())
            return 4;
        route_call_t kept(data_type_t::bf16, data_type_t::bf16,
                data_type_t::f32, 8, 96, 64, 4, 7000);
        kept.c.params.weight_cache_type = 1;
        if (nt::run_matmul(kept.c) != status_t::success
                || nt::native_layouts(kept.c.wei) != 0)
            return 5;
        return 0;
    }, "ZENDNNL_MATMUL_ALGO=11, ZENDNNL_BMM_ALGO=11");
    nt::expect_in_child(
            {{"ZENDNNL_MATMUL_ALGO", "10"}, {"ZENDNNL_BMM_ALGO", "10"}},
            [native_isa] {
        matmul_config_t &cfg = matmul_config_t::instance();
        const int32_t native
                = static_cast<int32_t>(matmul_algo_t::native_brgemm);
        if (cfg.get_algo() != native || cfg.get_bmm_algo() != native
                || !matmul_config_t::is_native_brgemm_algo()) {
            std::fprintf(stderr, "algo %d, bmm algo %d\n", cfg.get_algo(),
                    cfg.get_bmm_algo());
            return 1;
        }
        if (mm::resolve_matmul_backend() != group_unset_kernel()) return 2;
        if (!native_isa) return 0;
        route_call_t r(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
                8, 96, 64, 4, 6800);
        r.c.params.weight_cache_type = 1;
        nt::call_t asked = r.c;
        asked.dst = r.ref.get();
        asked.params.lowoha_algo = matmul_algo_t::native_brgemm;
        if (nt::run_matmul(r.c) != status_t::success
                || nt::native_layouts(r.c.wei) < 1
                || nt::run_matmul(asked) != status_t::success || !r.same_bits())
            return 3;
        return 0;
    }, "ZENDNNL_MATMUL_ALGO=10, ZENDNNL_BMM_ALGO=10");
    nt::expect_in_child(
            {{"ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES", "12,10,1,11,13,2"}}, [] {
        const std::vector<matmul_algo_t> want
                = {matmul_algo_t::aocl_dlp_blocked,
                        matmul_algo_t::moe_custom_kernel,
                        matmul_algo_t::onednn_blocked};
        return mm::get_algo_candidates() == want ? 0 : 1;
    }, "ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES=12,10,1,11,13,2");
}

} // namespace
