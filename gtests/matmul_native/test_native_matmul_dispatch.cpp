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
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
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
/// pointer-keyed caches of the GEMV packs.
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

    /// The descriptor native_layout_matmul builds for the call.
    nat::GemmDescriptor descriptor() const {
        nat::GemmDescriptor d;
        d.M = c.M;
        d.N = c.N;
        d.K = c.K;
        d.lda = c.lda_or_tight();
        d.ldb = c.ldb_or_tight();
        d.ldc = c.ldc_or_tight();
        d.transA = c.transA;
        d.transB = c.transB;
        d.alpha = c.alpha;
        d.beta = c.beta;
        d.src_dt = c.params.dtypes.src;
        d.wei_dt = c.params.dtypes.wei;
        d.dst_dt = c.params.dtypes.dst;
        d.bias_dt = c.params.dtypes.bias;
        d.bias = c.bias;
        d.is_weights_const = c.is_weights_const;
        d.num_threads = c.params.num_threads;
        d.gated_act = c.gated;
        return d;
    }

    /// The GEMV plan choice of the call (decide_native_gemv).
    nat::native_gemv_plan_t gemv() const {
        return nat::decide_native_gemv(descriptor(), c.params);
    }

    /// The GEMV kernel whose gates take the call: bf16_gemv_choice on the
    /// team decide_native_gemv gives it, or int8_gemv_choice on one thread.
    nat::native_gemv_plan_t gates() const {
        nat::GemmDescriptor d = descriptor();
        const nat::UarchParams &u = nat::detect_uarch();
        if (d.src_dt != data_type_t::bf16) {
            d.num_threads = 1;
            return nat::int8_gemv_choice(d, u, c.params);
        }
        if (d.num_threads > 1)
            d.num_threads = nat::m1_gemv_cap_threads(d.N, d.num_threads);
        return nat::bf16_gemv_choice(d, u, c.params);
    }

    /// dst and ref hold identical bits.
    ::testing::AssertionResult same_bits() const {
        return nt::within_steps(dst.get(), c.N, ref.get(), c.N,
                c.params.dtypes.dst, c.M, c.N, 0);
    }
};

// Which GEMV kernel's gates take a call, rule by rule: M = 1 only; bf16 on a
// team of m1_gemv_cap_threads of one thread (narrow at N <= 4 on a [K, N]
// weight, else the BKC pack, declining a pack over the L2 for a weight that
// is not constant), none on a team of more; INT8 on one thread at any team,
// with its pack within the L2, ldc == N, alpha 1 and beta 0, a per-tensor
// f32 source scale, f32 weight scales per tensor or per column, and no
// weight zero point or dst scale; at most one fusable post-op, a bf16 / f32
// bias and dst, ldc >= N, no transposed source.
TEST(NativeMatmulDispatch, GemvKernelGatesFollowTheGemvRules) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const size_t l2 = static_cast<size_t>(nat::detect_uarch().l2_bytes);
    const data_type_t bf16 = data_type_t::bf16, s8 = data_type_t::s8,
                      u8 = data_type_t::u8, f32 = data_type_t::f32;
    using g = nat::native_gemv_t;
    // K of a bf16 weight of 128 columns whose BKC pack exceeds the L2.
    const int k_over_l2 = static_cast<int>(l2 / (128 * 2)) + 64;
    uint32_t seed = 5100;
    const auto check = [&](const char *name, const route_call_t &r, g kernel) {
        SCOPED_TRACE(name);
        const nat::native_gemv_plan_t p = r.gates();
        EXPECT_EQ(static_cast<int>(p.kernel), static_cast<int>(kernel))
                << nat::native_gemv_name(p.kernel);
    };
    {
        route_call_t r(bf16, bf16, f32, 1, 3, 200, 1, seed += 4);
        check("bf16 N 3", r, g::bf16_narrow);
        r.c.params.num_threads = 8;
        check("bf16 N 3, 8 threads: a team of one", r, g::bf16_narrow);
        r.c.transB = true;
        check("bf16 N 3 transB", r, g::bf16_bkc);
    }
    {
        route_call_t r(bf16, bf16, bf16, 1, 100, 128, 1, seed += 4);
        check("bf16 N 100", r, g::bf16_bkc);
        r.c.params.num_threads = 8;
        check("bf16 N 100, 8 threads: a team of two", r, g::none);
        r.c.M = 2;
        check("bf16 M 2", r, g::none);
        r.c.M = 1;
        r.c.transA = true;
        check("bf16 transA", r, g::none);
        r.c.transA = false;
        r.c.ldc = 64;
        check("bf16 ldc below N", r, g::none);
        r.c.ldc = 0;
        r.c.params.dtypes.dst = s8;
        check("bf16 to s8", r, g::none);
    }
    {
        route_call_t r(bf16, bf16, f32, 1, 64, 128, 8, seed += 4);
        check("bf16 N 64, 8 threads: a team of one", r, g::bf16_bkc);
        r.c.params.num_threads = 2;
        r.c.N = 256;
        check("bf16 N 256, 2 threads", r, g::none);
    }
    {
        route_call_t r(bf16, bf16, f32, 1, 128, k_over_l2, 1, seed += 4);
        check("bf16 pack over the L2, constant", r, g::bf16_bkc);
        r.c.is_weights_const = false;
        check("bf16 pack over the L2, not constant", r, g::none);
    }
    {
        route_call_t r(bf16, bf16, f32, 1, 100, 128, 1, seed += 4);
        mm::matmul_post_op relu;
        relu.po_type = post_op_type_t::relu;
        relu.alpha = 0.0f;
        r.c.params.postop_ = {relu};
        check("bf16 relu", r, g::bf16_bkc);
        EXPECT_EQ(static_cast<int>(r.gates().fused),
                static_cast<int>(nat::fused_postop_t::relu));
        mm::matmul_post_op sigmoid;
        sigmoid.po_type = post_op_type_t::sigmoid;
        r.c.params.postop_ = {relu, sigmoid};
        check("bf16 relu then sigmoid", r, g::none);
        mm::matmul_post_op add;
        add.po_type = post_op_type_t::binary_add;
        r.c.params.postop_ = {add};
        check("bf16 binary_add", r, g::none);
        r.c.params.postop_.clear();
        const nt::buffer_t bias
                = nt::random_buffer(data_type_t::f16, 100, 9, -1.0f, 1.0f);
        r.c.bias = bias.get();
        r.c.params.dtypes.bias = data_type_t::f16;
        check("bf16 f16 bias", r, g::none);
        r.c.params.dtypes.bias = bf16;
        check("bf16 bf16 bias", r, g::bf16_bkc);
    }
    {
        route_call_t r(u8, s8, f32, 1, 96, 64, 1, seed += 4);
        check("u8 x s8", r, g::int8_bkc);
        r.wei_scale_cols(96);
        check("u8 x s8, weight scales of N", r, g::int8_bkc);
        r.c.params.quant_params.wei_scale.dt = bf16;
        check("u8 x s8, bf16 weight scales of N", r, g::none);
        r.c.params.quant_params.wei_scale.dt = data_type_t::f16;
        check("u8 x s8, f16 weight scales of N", r, g::none);
        r.c.params.quant_params.wei_scale.dt = f32;
        r.wei_scale_cols(7);
        check("u8 x s8, 7 weight scales", r, g::none);
        r.wei_scale_cols(96);
        r.c.params.quant_params.wei_scale.dims = {2, 96};
        check("u8 x s8, per-group weight scales", r, g::none);
        r.c.params.quant_params.wei_scale.dims = {1, 96};
        r.c.params.quant_params.src_scale.dims = {1, 4};
        check("u8 x s8, 4 source scales", r, g::none);
        r.c.params.quant_params.src_scale.dims = {1, 1};
        r.c.params.quant_params.src_scale.dt = bf16;
        check("u8 x s8, a bf16 source scale", r, g::none);
        r.c.params.quant_params.src_scale.dt = f32;
        const int32_t zp = 3;
        auto &q = r.c.params.quant_params;
        q.src_zp.buff = &zp;
        q.src_zp.dt = data_type_t::s32;
        q.src_zp.dims = {1};
        check("u8 x s8, an s32 source zero point", r, g::int8_bkc);
        q.src_zp.dt = f32;
        check("u8 x s8, an f32 source zero point", r, g::none);
        q.src_zp.buff = nullptr;
        q.wei_zp.buff = &zp;
        q.wei_zp.dt = data_type_t::s32;
        q.wei_zp.dims = {1};
        check("u8 x s8, a weight zero point", r, g::none);
        q.wei_zp.buff = nullptr;
        const float dst_scale = 2.0f;
        q.dst_scale.buff = &dst_scale;
        q.dst_scale.dt = f32;
        q.dst_scale.dims = {1};
        check("u8 x s8, a dst scale", r, g::none);
        q.dst_scale.buff = nullptr;
        r.c.params.num_threads = 2;
        check("u8 x s8, 2 threads: on one thread", r, g::int8_bkc);
        r.c.params.num_threads = 1;
        r.c.ldc = 112;
        check("u8 x s8, ldc above N", r, g::none);
        r.c.ldc = 0;
        r.c.alpha = 2.0f;
        check("u8 x s8, alpha 2", r, g::none);
        r.c.alpha = 1.0f;
        r.c.beta = 0.5f;
        check("u8 x s8, beta 0.5", r, g::none);
        r.c.beta = 0.0f;
        r.c.params.dtypes.dst = bf16;
        check("u8 x s8 to bf16", r, g::int8_bkc);
        r.c.params.dtypes.dst = s8;
        check("u8 x s8 to s8", r, g::none);
    }
    {
        route_call_t r(s8, s8, f32, 1, 4096, 4096, 1, seed += 4);
        check("s8 x s8 pack over the L2", r, g::none);
        r.c.N = 96;
        r.c.K = 64;
        check("s8 x s8", r, g::int8_bkc);
        r.c.params.quant_params.wei_scale.dt = data_type_t::f16;
        check("s8 x s8, an f16 weight scale", r, g::none);
    }
    {
        route_call_t r(f32, f32, f32, 1, 96, 64, 1, seed += 4);
        check("f32", r, g::none);
    }
}

// The GEMV plan choice of a call that asks for the native BRGEMM at M = 1:
// the narrow kernel for N <= 4 on a [K, N] weight, the only native kernel
// for those widths, at any team; none with a gated activation, alpha 0 or a
// Swish of alpha other than 1, which the GEMV kernels do not apply, and
// none at M > 1.  The BKC kernels take N a multiple of 32 and a constant
// weight within the rule's bounds: bf16_bkc on one thread, or on a team of
// one at K <= 1024; int8_bkc at N * K <= 64 Ki at any team, or on one
// thread at K <= 1024.
TEST(NativeMatmulDispatch, GemvPlanChoiceOfRequestedCalls) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32,
                      u8 = data_type_t::u8, s8 = data_type_t::s8;
    const auto check = [](const char *name, const route_call_t &r, g kernel) {
        SCOPED_TRACE(name);
        EXPECT_EQ(static_cast<int>(r.gemv().kernel), static_cast<int>(kernel))
                << nat::native_gemv_name(r.gemv().kernel);
    };
    uint32_t seed = 6400;
    for (const int threads : {1, 8, 32}) {
        SCOPED_TRACE("threads " + std::to_string(threads));
        route_call_t r(bf16, bf16, f32, 1, 3, 200, threads, seed += 4);
        check("N 3", r, g::bf16_narrow);
        mm::matmul_post_op swish;
        swish.po_type = post_op_type_t::swish;
        swish.alpha = 1.0f;
        r.c.params.postop_ = {swish};
        check("N 3, SiLU", r, g::bf16_narrow);
        r.c.params.postop_[0].alpha = 0.5f;
        check("N 3, Swish of alpha 0.5", r, g::none);
        r.c.params.postop_.clear();
        r.c.alpha = 0.0f;
        check("N 3, alpha 0", r, g::none);
        r.c.alpha = 1.0f;
        r.c.M = 2;
        check("N 3, M 2", r, g::none);
        r.c.M = 1;
        r.c.transB = true;
        check("N 3, an [N, K] weight", r, g::none);
    }
    {
        route_call_t r(bf16, bf16, f32, 1, 4, 64, 1, seed += 4);
        r.c.gated = mm::matmul_gated_act_t::silu_and_mul;
        check("N 4, a gated activation", r, g::none);
    }
    {
        route_call_t r(bf16, bf16, f32, 1, 100, 128, 1, seed += 4);
        ASSERT_EQ(static_cast<int>(r.gates().kernel),
                static_cast<int>(g::bf16_bkc));
        check("N 100", r, g::none);
        r.c.N = 96;
        check("N 96", r, g::bf16_bkc);
        r.c.is_weights_const = false;
        check("N 96, a weight that is not constant", r, g::none);
        r.c.is_weights_const = true;
        r.c.params.num_threads = 8;
        check("N 96, 8 threads: a team of two", r, g::none);
        r.c.N = 64;
        r.c.K = 1024;
        check("N 64 K 1024, 8 threads: a team of one", r, g::bf16_bkc);
        r.c.K = 1056;
        check("N 64 K 1056, 8 threads", r, g::none);
        r.c.params.num_threads = 1;
        r.c.K = 4096;
        check("N 64 K 4096", r, g::bf16_bkc);
    }
    {
        route_call_t r(u8, s8, f32, 1, 64, 1024, 8, seed += 4);
        check("u8 N 64 K 1024, 8 threads", r, g::int8_bkc);
        r.c.N = 96;
        check("u8 N 96 K 1024, 8 threads", r, g::none);
        r.c.params.num_threads = 1;
        check("u8 N 96 K 1024", r, g::int8_bkc);
        r.c.K = 1056;
        check("u8 N 96 K 1056", r, g::none);
        r.c.N = 32;
        r.c.K = 2048;
        check("u8 N 32 K 2048", r, g::int8_bkc);
        r.c.is_weights_const = false;
        check("u8 N 32 K 2048, a weight that is not constant", r, g::none);
    }
    {
        route_call_t r(s8, s8, f32, 1, 96, 64, 32, seed += 4);
        check("s8 N 96 K 64, 32 threads", r, g::int8_bkc);
        r.c.N = 100;
        check("s8 N 100 K 64, 32 threads", r, g::none);
    }
}

// A call that asks for the native BRGEMM at M = 1 with N <= 4 on a [K, N]
// weight runs on the narrow kernel, per call and under ZENDNNL_MATMUL_ALGO=10:
// the bits of the kernel run directly, and no native layout of the weight.
TEST(NativeMatmulDispatch, RequestedNarrowCallsTakeTheNarrowKernel) {
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
            const nat::native_gemv_plan_t plan = r.gemv();
            ASSERT_EQ(static_cast<int>(plan.kernel),
                    static_cast<int>(nat::native_gemv_t::bf16_narrow));
            std::memcpy(r.ref.get(), r.dst.get(), r.dst.bytes);
            ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
            ASSERT_EQ(nat::run_native_gemv(plan, r.descriptor(), r.c.src,
                              r.c.wei, r.ref.get(), r.c.bias, r.c.params, 1),
                    status_t::success);
            EXPECT_TRUE(r.same_bits());
            EXPECT_EQ(nt::native_layouts(r.c.wei), 0u);
        }
}

// A call that asks for the native BRGEMM at M = 1 whose GEMV plan choice is a
// BKC kernel runs on it, per call and under ZENDNNL_MATMUL_ALGO=10: the bits
// of the kernel run directly, within the band of the reference, and no
// native layout of the weight.
TEST(NativeMatmulDispatch, RequestedBkcCallsTakeTheBkcKernels) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    using g = nat::native_gemv_t;
    const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32,
                      u8 = data_type_t::u8, s8 = data_type_t::s8;
    struct case_t {
        data_type_t src, dst;
        int N, K, threads;
        bool bias, relu;
        g kernel;
    };
    const case_t cases[] = {
            {bf16, bf16, 96, 300, 1, false, false, g::bf16_bkc},
            {bf16, f32, 64, 512, 8, true, true, g::bf16_bkc},
            {u8, f32, 128, 512, 1, true, false, g::int8_bkc},
            {s8, bf16, 64, 256, 8, false, true, g::int8_bkc},
    };
    uint32_t seed = 6700;
    for (const bool per_call : {true, false})
        for (const case_t &cs : cases) {
            SCOPED_TRACE(std::string(per_call ? "lowoha_algo native_brgemm"
                                              : "ZENDNNL_MATMUL_ALGO=10")
                    + " " + nat::native_gemv_name(cs.kernel) + " N "
                    + std::to_string(cs.N) + " K " + std::to_string(cs.K)
                    + " threads " + std::to_string(cs.threads));
            std::unique_ptr<nt::algo_pin_t> pin;
            if (!per_call)
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            const bool int8 = cs.src != bf16;
            route_call_t r(cs.src, int8 ? s8 : bf16, cs.dst, 1, cs.N, cs.K,
                    cs.threads, seed += 4);
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
            const nat::native_gemv_plan_t plan = r.gemv();
            ASSERT_EQ(
                    static_cast<int>(plan.kernel), static_cast<int>(cs.kernel));
            ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
            ASSERT_EQ(nat::run_native_gemv(plan, r.descriptor(), r.c.src,
                              r.c.wei, r.ref.get(), r.c.bias, r.c.params, 1),
                    status_t::success);
            EXPECT_TRUE(r.same_bits());
            EXPECT_EQ(nt::native_layouts(r.c.wei), 0u);
            ASSERT_EQ(nt::run_reference(r.c, r.c.wei, r.ref.get()),
                    status_t::success);
            EXPECT_TRUE(nt::within_band(r.dst.get(), cs.N, r.ref.get(), cs.N,
                    cs.dst, 1, cs.N, rtol_bf16,
                    nt::comparison_abs_bound(cs.K, epsilon_bf16, 1.0f, 0.0f,
                            true, int8, false)));
        }
}

// A u8 / s8 call that asks for the native BRGEMM at M = 1, of a shape whose
// GEMV plan choice is int8_bkc with f32 weight scales, gets no GEMV choice
// with bf16 or f16 weight scales (per tensor or of N elements), which the
// INT8 GEMV kernel reads as f32 only, per call and under
// ZENDNNL_MATMUL_ALGO=10: the native tiles serve it, with a layout cached
// for its weight, and give the reference's output within the INT8 band.
TEST(NativeMatmulDispatch, RequestedInt8CallsWithF16OrBf16ScalesTakeTheTiles) {
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
                    ASSERT_EQ(static_cast<int>(r.gemv().kernel),
                            static_cast<int>(nat::native_gemv_t::int8_bkc))
                            << "with f32 weight scales";
                    const nt::buffer_t wei_scales = nt::random_buffer(scale_dt,
                            static_cast<size_t>(scales), seed + 2, 0.125f,
                            0.5f);
                    auto &ws = r.c.params.quant_params.wei_scale;
                    ws.buff = wei_scales.get();
                    ws.dt = scale_dt;
                    ws.dims = {1, scales};
                    EXPECT_EQ(static_cast<int>(r.gemv().kernel),
                            static_cast<int>(nat::native_gemv_t::none));
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
// weight, of a shape whose GEMV plan choice is int8_bkc at alpha 1 and beta
// 0, gets no GEMV choice with alpha other than 1 or beta other than 0, which
// the INT8 GEMV kernel applies around its dequantized value, not in the
// reference order, per call and under ZENDNNL_MATMUL_ALGO=10: the native
// tiles serve it, with a layout cached for its weight and the bits of the
// same call on a weight that is not constant, and give the reference's
// output within the INT8 band.  The calls carry a bias, source and weight
// scales other than 1, a dst that beta reads, and with a u8 source a zero
// point.
TEST(NativeMatmulDispatch, RequestedInt8CallsWithAlphaOrBetaTakeTheTiles) {
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
                    ASSERT_EQ(static_cast<int>(r.gemv().kernel),
                            static_cast<int>(nat::native_gemv_t::int8_bkc))
                            << "at alpha 1 and beta 0";
                    r.c.alpha = ab.alpha;
                    r.c.beta = ab.beta;
                    EXPECT_EQ(static_cast<int>(r.gemv().kernel),
                            static_cast<int>(nat::native_gemv_t::none));
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
// with no GEMV plan choice runs on the native tiles, which cache a layout of
// its weight.
TEST(NativeMatmulDispatch, RequestedCallsTakeTheNativeTiles) {
    if (!gemv_isa()) GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    route_call_t asked(data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
            1, 96, 128, 8, 5800);
    asked.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
    asked.c.params.weight_cache_type = 1;
    ASSERT_EQ(static_cast<int>(asked.gemv().kernel),
            static_cast<int>(nat::native_gemv_t::none));
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
