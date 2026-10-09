/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// CK -> DLP fallback chain — when `prepare_for_call` refuses, the
/// dispatcher must route to the DLP path cleanly so the caller's
/// output is correct (just slower).  These tests pin that chain.
///
/// Strategy:
///   1. Build a shape that auto-select would route to ALGO 3 (decode-
///      class) BUT make the dtype tuple ineligible for CK (e.g.,
///      `dst_dt = f16` which CK refuses).
///   2. Run `group_matmul_direct` and verify the output matches the
///      same call run on a known-DLP path (forced ALGO=1).  If the
///      fallback chain is broken — kernel runs anyway and produces
///      garbage, or assert-fails in `dispatch_tile` — this test will
///      catch it.
///
/// The "rejected dtype that auto-select would otherwise route to
/// ALGO 3" cases exercised here are:
///   * dst = f16 (refused by `resolve_variant`).
///   * is_weights_const = false (refused by `prepare_for_call`).
///
/// f32 dst is supported by the kernel today, so it cannot serve as
/// the rejected-dtype fallback case — f16 dst is used instead.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <type_traits>

#include "ck_test_helpers.hpp"
#include "common/op_config.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/matmul_route.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_custom_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mt = moe_test_utils;
namespace ck = ck_test::ck;
using mt::bfloat16_t;
using mt::data_type_t;
using mt::group_matmul_direct;
using mt::matmul_params;
using mt::status_t;

struct ScalarRouteConfigGuard {
    zendnnl::common::matmul_config_t &config
            = zendnnl::common::matmul_config_t::instance();
    int32_t old_algo = config.get_algo();
    int32_t old_cache = config.get_weight_cache();
    bool old_route = config.get_custom_kernel_route();
    int old_dynamic = omp_get_dynamic();

    ~ScalarRouteConfigGuard() {
        config.set_algo(old_algo);
        config.set_weight_cache(old_cache);
        config.set_custom_kernel_route(old_route);
        omp_set_dynamic(old_dynamic);
    }
};

struct ScalarW4Problem {
    int M;
    int N;
    int K;
    int group_size;
    int G;
    int num_threads;

    std::vector<bfloat16_t> src;
    std::vector<int8_t> weight;
    std::vector<float> src_scale;
    std::vector<float> wei_scale;
    matmul_params params;

    ScalarW4Problem(int rows, int output_cols, int reduction, int scale_group,
            int threads)
        : M(rows)
        , N(output_cols)
        , K(reduction)
        , group_size(scale_group)
        , G(K / group_size)
        , num_threads(threads)
        , src(static_cast<size_t>(M) * K)
        , weight((static_cast<size_t>(N) * K + 1) / 2, 0)
        , src_scale(M, 0.0f)
        , wei_scale(static_cast<size_t>(G) * N) {
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = bfloat16_t((static_cast<int>(i % 19) - 9) * 0.03125f);
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const int value
                        = static_cast<int>(
                                  (static_cast<uint32_t>(k) * 3U
                                          + static_cast<uint32_t>(n) * 5U)
                                  % 15U)
                        - 7;
                const size_t index = static_cast<size_t>(n) * K + k;
                uint8_t byte = static_cast<uint8_t>(weight[index >> 1]);
                const uint8_t nibble
                        = static_cast<uint8_t>(value) & uint8_t {0x0F};
                byte = (index & 1U)
                        ? static_cast<uint8_t>((byte & 0x0FU) | (nibble << 4))
                        : static_cast<uint8_t>((byte & 0xF0U) | nibble);
                weight[index >> 1] = static_cast<int8_t>(byte);
            }
        }
        for (int g = 0; g < G; ++g) {
            for (int n = 0; n < N; ++n) {
                wei_scale[static_cast<size_t>(g) * N + n]
                        = 0.004f + 0.00025f * static_cast<float>((g + n) % 7);
            }
        }

        params.dtypes.src = data_type_t::bf16;
        params.dtypes.wei = data_type_t::s4;
        params.dtypes.dst = data_type_t::bf16;
        params.dtypes.bias = data_type_t::none;
        params.dtypes.compute = data_type_t::s8;
        params.dynamic_quant = true;
        params.mem_format_a = 'n';
        params.mem_format_b = 'n';
        params.weight_cache_type = 2;
        params.num_threads = num_threads;
        params.quant_params.src_scale.buff = src_scale.data();
        params.quant_params.src_scale.dt = data_type_t::f32;
        params.quant_params.src_scale.dims = {M, 1};
        params.quant_params.wei_scale.buff = wei_scale.data();
        params.quant_params.wei_scale.dt = data_type_t::f32;
        params.quant_params.wei_scale.dims = {G, N};
    }

    status_t run(std::vector<bfloat16_t> &dst) {
        zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;
        return zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                1.0f, src.data(), K, weight.data(), K, /*bias=*/nullptr, 0.0f,
                dst.data(), N, /*is_weights_const=*/true, batch_params, params);
    }
};

struct ScalarW4RouteShape {
    int M;
    int N;
    int K;
    int group_size;
    int num_threads;
};

std::vector<ScalarW4RouteShape> make_scalar_w4_route_stress_shapes() {
    constexpr int kCases = 400;
    constexpr int kThreadCounts[] = {2, 4, 8, 16, 32, 64};
    constexpr int kGroupSizes[] = {8, 16, 32, 64};
    std::mt19937 rng(0xD34A8U);
    std::vector<ScalarW4RouteShape> shapes;
    shapes.reserve(kCases);
    for (int i = 0; i < kCases; ++i) {
        const int num_threads = kThreadCounts[i % 6];
        const int group_size = kGroupSizes[(i / 6) % 4];
        const int groups = 2 + static_cast<int>(rng() % 3);
        const int K = group_size * groups;
        const int extra_n_tiles
                = (i % 5 == 0) ? 0 : static_cast<int>(rng() % 5);
        const int N = custom_kernel::kNRMin * (2 * num_threads + extra_n_tiles);
        const int M = (i % 2 == 0) ? 1 + static_cast<int>(rng() % 32)
                                   : 33 + static_cast<int>(rng() % 96);
        shapes.push_back({M, N, K, group_size, num_threads});
    }
    return shapes;
}

class CkDispatchRoutingW4EndToEnd
    : public ::testing::TestWithParam<ScalarW4RouteShape> {};

std::string scalar_w4_route_shape_name(
        const ::testing::TestParamInfo<ScalarW4RouteShape> &info) {
    const auto &s = info.param;
    return "Case" + std::to_string(info.index) + "_M" + std::to_string(s.M)
            + "_N" + std::to_string(s.N) + "_K" + std::to_string(s.K) + "_G"
            + std::to_string(s.group_size) + "_T"
            + std::to_string(s.num_threads);
}

TEST(CkDispatchRouting, ScalarRouteUsesEffectiveMatmulAlgo1) {
    CK_SKIP_IF_NO_BF16_ISA();

    ScalarRouteConfigGuard guard;
    mt::AlgoEnvGuard auto_group_algo(0);
    mt::CustomKernelOverride group_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;

    const auto routable = [&]() {
        return zendnnl::lowoha::matmul::custom_kernel_routable('r',
                /*M=*/8, /*N=*/2048, /*K=*/256, /*num_threads=*/32,
                /*batch_count=*/1, /*transA=*/false, /*transB=*/true,
                /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/nullptr, /*ldb=*/256,
                /*is_weights_const=*/true, params);
    };

    // The effective grouped fallback is AOCL-DLP blocked.
    EXPECT_TRUE(routable());

    // A per-operation API selection remains stronger than the process-wide
    // CK policy because grouped fallback cannot preserve that local choice.
    params.lowoha_algo = zendnnl::common::matmul_algo_t::onednn;
    EXPECT_FALSE(routable());

    // Disabling the scalar CK knob restores the configured AOCL path.
    params.lowoha_algo = zendnnl::common::matmul_algo_t::none;
    guard.config.set_custom_kernel_route(false);
    EXPECT_FALSE(routable());
}

TEST(CkDispatchRouting, ScalarRouteUsesRequestedThreadTileCapacity) {
    CK_SKIP_IF_NO_BF16_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;

    struct RouteCase {
        int N;
        int num_threads;
    };
    const RouteCase boundary_cases[] = {
            {128, 2}, {256, 4}, {512, 8}, {1024, 16}, {2048, 32}, {4096, 64}};
    for (const auto &c : boundary_cases) {
        EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r',
                /*M=*/8, c.N, /*K=*/256, c.num_threads, /*batch_count=*/1,
                /*transA=*/false, /*transB=*/true, /*alpha=*/1.0f,
                /*beta=*/0.0f, /*bias=*/nullptr, /*ldb=*/256,
                /*is_weights_const=*/true, params))
                << "boundary route rejected for N=" << c.N
                << " num_threads=" << c.num_threads;
        EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r',
                /*M=*/8, c.N, /*K=*/256, c.num_threads + 1,
                /*batch_count=*/1, /*transA=*/false, /*transB=*/true,
                /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/nullptr, /*ldb=*/256,
                /*is_weights_const=*/true, params))
                << "under-capacity route accepted for N=" << c.N
                << " num_threads=" << (c.num_threads + 1);
    }
}

TEST(CkDispatchRouting, ScalarW4RouteUsesThreadAndGroupBoundaries) {
    CK_SKIP_IF_NO_INT8_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    float src_scale = 1.0f;
    float wei_scale = 1.0f;
    const auto make_params = [&](int M, int N, int K, int group_size) {
        matmul_params params;
        params.dtypes.src = data_type_t::bf16;
        params.dtypes.wei = data_type_t::s4;
        params.dtypes.dst = data_type_t::bf16;
        params.dtypes.bias = data_type_t::none;
        params.dtypes.compute = data_type_t::s8;
        params.dynamic_quant = true;
        params.mem_format_a = 'n';
        params.mem_format_b = 'n';
        params.weight_cache_type = 2;
        params.quant_params.src_scale.buff = &src_scale;
        params.quant_params.src_scale.dt = data_type_t::f32;
        params.quant_params.src_scale.dims = {M, 1};
        params.quant_params.wei_scale.buff = &wei_scale;
        params.quant_params.wei_scale.dt = data_type_t::f32;
        params.quant_params.wei_scale.dims = {K / group_size, N};
        return params;
    };
    const auto routable
            = [&](int M, int N, int K, int group_size, int num_threads) {
        const matmul_params params = make_params(M, N, K, group_size);
        return zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
                num_threads, 1, false, true, 1.0f, 0.0f, nullptr, K, true,
                params);
    };

    constexpr int M = 8;
    constexpr int K = 64;
    constexpr int group_size = 32;
    for (const int num_threads : {2, 4, 8, 16, 32, 64}) {
        const int min_n = 2 * num_threads * custom_kernel::kNRMin;
        EXPECT_TRUE(routable(M, min_n, K, group_size, num_threads))
                << "exact W4 tile capacity rejected for threads="
                << num_threads;
        EXPECT_FALSE(routable(
                M, min_n - custom_kernel::kNRMin, K, group_size, num_threads))
                << "aligned but under-capacity W4 route accepted for threads="
                << num_threads;
        EXPECT_FALSE(routable(M, min_n + 1, K, group_size, num_threads))
                << "non-32-aligned W4 N accepted for threads=" << num_threads;
    }

    for (const int candidate_group_size : {8, 16, 32, 64}) {
        const int candidate_k = 2 * candidate_group_size;
        EXPECT_TRUE(routable(
                M, /*N=*/128, candidate_k, candidate_group_size, /*threads=*/2))
                << "valid W4 group size rejected: " << candidate_group_size;
    }
}

TEST(CkDispatchRouting, ScalarRouteAcceptsInt8AndW4) {
    CK_SKIP_IF_NO_INT8_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    constexpr int M = 8;
    constexpr int N = 2048;
    constexpr int K = 256;
    float src_scale = 1.0f;
    float wei_scale = 1.0f;

    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    {
        mt::CustomKernelInt8Override int8_ck_on(true);
        EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
                K, 32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    }
    {
        mt::CustomKernelInt8Override int8_ck_off(false);
        EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
                K, 32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    }

    params.dtypes.wei = data_type_t::s4;
    params.quant_params.wei_scale.dims = {K / 32, N};
    EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    // W4A8 has exactly two supported source forms: dynamic BF16 and static,
    // already-quantized S8. S8 with dynamic_quant=true is contradictory and
    // must not be classified or routed as a W4A8 custom-kernel call.
    params.dtypes.src = data_type_t::s8;
    params.dynamic_quant = false;
    params.quant_params.src_scale.buff = &src_scale;
    EXPECT_TRUE(zendnnl::lowoha::matmul::is_w4a8_config(params));
    EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    params.dynamic_quant = true;
    EXPECT_FALSE(zendnnl::lowoha::matmul::is_w4a8_config(params));
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    params.dtypes.src = data_type_t::bf16;
    params.dynamic_quant = true;

    // Group N-tile requires true per-group {G>1,N} metadata.
    params.quant_params.wei_scale.dims = {1, N};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    // The S4 pack has no N-tail kernel.
    params.quant_params.wei_scale.dims = {K / 32, N + 1};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N + 1,
            K, 32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    // The S4 custom kernel consumes whole K-octets per scale group.
    params.quant_params.wei_scale.dims = {K / 4, N};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    params.quant_params.wei_scale.dims = {K / 32, N - 1};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    int32_t src_zp = 0;
    params.quant_params.wei_scale.dims = {K / 32, N};
    params.quant_params.src_zp.buff = &src_zp;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
}

TEST(CkDispatchRouting, ScalarRouteAcceptsStaticU8S8PerTensor) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    CK_SKIP_IF_NO_INT8_ISA();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int M = 5, N = 256, K = 64;
    float src_scale = 0.02f, wei_scale = 0.03f, dst_scale = 2.0f;
    std::vector<float> wei_scales(N, wei_scale);
    int32_t src_zp = 127, dst_zp = 119;
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::none;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, 1};

    const std::vector<std::vector<int64_t>> scalar_shapes {{}, {1}, {1, 1}};
    for (const auto &shape : scalar_shapes) {
        params.quant_params.src_scale.dims = shape;
        params.quant_params.src_zp.dims = shape;
        params.quant_params.wei_scale.dims = shape;
        for (auto dst :
                {data_type_t::bf16, data_type_t::f32, data_type_t::u8}) {
            params.dtypes.dst = dst;
            params.quant_params.dst_scale = {};
            params.quant_params.dst_zp = {};
            if (dst == data_type_t::u8) {
                params.quant_params.dst_scale.buff = &dst_scale;
                params.quant_params.dst_scale.dt = data_type_t::f32;
                params.quant_params.dst_scale.dims = shape;
                params.quant_params.dst_zp.buff = &dst_zp;
                params.quant_params.dst_zp.dt = data_type_t::s32;
                params.quant_params.dst_zp.dims = shape;
            }
            EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M,
                    N, K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true,
                    params));
        }
    }
    params.dtypes.dst = data_type_t::u8;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.dims = {1, 1};
    params.quant_params.dst_scale.dims = {1, 1};
    params.quant_params.dst_zp.dims = {1, 1};

    int32_t wei_zp = 0;
    params.quant_params.wei_zp.buff = &wei_zp;
    params.quant_params.wei_zp.dt = data_type_t::s32;
    params.quant_params.wei_zp.dims = {1, 1};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.quant_params.wei_zp = {};

    params.quant_params.src_scale.dims = {M, 1};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.quant_params.src_scale.dims = {1, 1};

    params.quant_params.wei_scale.buff = wei_scales.data();
    for (const std::vector<int64_t> &shape :
            std::vector<std::vector<int64_t>> {{N}, {1, N}}) {
        params.quant_params.wei_scale.dims = shape;
        EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
                K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    }
    params.quant_params.wei_scale.dims = {N, 1};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dims = {1, 1};

    params.quant_params.dst_scale.dims = {1, N};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.quant_params.dst_scale.dims = {1, 1};

    params.quant_params.src_zp.buff = nullptr;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.quant_params.src_zp.buff = &src_zp;

    src_scale = -1.0f;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    src_scale = 0.02f;
    src_zp = 256;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    src_zp = 127;

    for (post_op_type_t supported :
            {post_op_type_t::relu, post_op_type_t::sigmoid}) {
        matmul_post_op po;
        po.po_type = supported;
        params.postop_ = {po};
        EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
                K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    }
    matmul_post_op unsupported;
    unsupported.po_type = post_op_type_t::gelu_tanh;
    params.postop_ = {unsupported};
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    params.postop_.clear();

    constexpr int unsafe_k = ((custom_kernel::kMaxSafeInt8AccumK / 4) + 1) * 4;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
            unsafe_k, 4, 1, false, true, 1.0f, 0.0f, nullptr, unsafe_k, true,
            params));
}

void check_scalar_w4_route_matches_fallback(const ScalarW4RouteShape &s) {
    ::reset_grp_matmul_caches();

    SCOPED_TRACE(::testing::Message()
            << "M=" << s.M << " N=" << s.N << " K=" << s.K << " group_size="
            << s.group_size << " num_threads=" << s.num_threads);

    ScalarRouteConfigGuard guard;
    mt::AlgoEnvGuard group_auto(0);
    mt::CustomKernelOverride group_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    // Keep the reference weights raw and independent from the route-on run.
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    ScalarW4Problem route_problem(s.M, s.N, s.K, s.group_size, s.num_threads);
    ScalarW4Problem fallback_problem(
            s.M, s.N, s.K, s.group_size, s.num_threads);
    ASSERT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', s.M, s.N,
            s.K, s.num_threads, 1, false, true, 1.0f, 0.0f, nullptr, s.K, true,
            route_problem.params));

    constexpr float kOutputPoison = -123.0f;
    std::vector<bfloat16_t> dst_route(
            static_cast<size_t>(route_problem.M) * route_problem.N,
            bfloat16_t(kOutputPoison));
    std::vector<bfloat16_t> dst_fallback(
            dst_route.size(), bfloat16_t(kOutputPoison));

    const char *mode = nullptr;
    {
        mt::GemmModeCaptureGuard capture;
        ASSERT_EQ(route_problem.run(dst_route), status_t::success);
        mode = zendnnl::lowoha::matmul::test_api ::
                       s_last_group_matmul_direct_gemm_mode.load(
                               std::memory_order_relaxed);
    }
    ASSERT_NE(mode, nullptr);
    if (s.M <= zendnnl::lowoha::matmul::kDecodeMaxM) {
        EXPECT_NE(std::strstr(mode, "custom"), nullptr)
                << "dense W4 decode did not execute CK; mode=" << mode;
    } else {
        EXPECT_NE(std::strstr(mode, "flat_m_tile"), nullptr)
                << "dense W4 prompt did not execute ALGO 2; mode=" << mode;
    }

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(fallback_problem.run(dst_fallback), status_t::success);

    size_t mismatch_count = 0;
    size_t unwritten_count = 0;
    size_t nonfinite_count = 0;
    size_t nonzero_count = 0;
    size_t first_mismatch = dst_route.size();
    float first_actual = 0.0f;
    float first_expected = 0.0f;
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < dst_route.size(); ++i) {
        const float actual = static_cast<float>(dst_route[i]);
        const float expected = static_cast<float>(dst_fallback[i]);
        const float abs_error = std::fabs(actual - expected);
        const float tolerance = 0.05f + 0.05f * std::fabs(expected);
        if (expected == kOutputPoison) { ++unwritten_count; }
        if (!std::isfinite(expected)) { ++nonfinite_count; }
        if (std::isfinite(expected) && std::fabs(expected) > 1.0e-6f) {
            ++nonzero_count;
        }
        if (abs_error > max_abs_error) { max_abs_error = abs_error; }
        if (!std::isfinite(actual) || !std::isfinite(expected)
                || abs_error > tolerance) {
            if (mismatch_count == 0) {
                first_mismatch = i;
                first_actual = actual;
                first_expected = expected;
            }
            ++mismatch_count;
        }
    }
    EXPECT_EQ(unwritten_count, 0u)
            << "scalar W4 fallback returned success without writing dst";
    EXPECT_EQ(nonfinite_count, 0u)
            << "scalar W4 fallback produced non-finite output";
    EXPECT_GT(nonzero_count, 0u)
            << "scalar W4 fallback produced an all-zero output";
    EXPECT_EQ(mismatch_count, 0u)
            << "dense W4 routed result differs; first_index=" << first_mismatch
            << " actual=" << first_actual << " expected=" << first_expected
            << " max_abs_error=" << max_abs_error;
}

TEST_P(CkDispatchRoutingW4EndToEnd, RouteMatchesFallbackAndUsesExpectedPhase) {
    CK_SKIP_IF_NO_INT8_ISA();
    check_scalar_w4_route_matches_fallback(GetParam());
}

INSTANTIATE_TEST_SUITE_P(DecodeAndPromptBoundaries, CkDispatchRoutingW4EndToEnd,
        ::testing::Values(ScalarW4RouteShape {/*M=*/8, /*N=*/512, /*K=*/64,
                                  /*group_size=*/32, /*num_threads=*/8},
                ScalarW4RouteShape {/*M=*/33, /*N=*/256, /*K=*/64,
                        /*group_size=*/32, /*num_threads=*/4}),
        scalar_w4_route_shape_name);

TEST(CkDispatchRoutingW4Stress, RandomDenseShapes) {
    CK_SKIP_IF_NO_INT8_ISA();
    const char *enabled = std::getenv("ZENDNNL_GTEST_W4_ROUTE_STRESS");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        GTEST_SKIP()
                << "set ZENDNNL_GTEST_W4_ROUTE_STRESS=1 for the 400-case sweep";
    }

    for (const auto &shape : make_scalar_w4_route_stress_shapes()) {
        check_scalar_w4_route_matches_fallback(shape);
        if (::testing::Test::HasFatalFailure()) { return; }
    }
}

TEST(CkDispatchRouting, ScalarInt8EndToEndMatchesFallbackAndRunsCk) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    constexpr int M = 8, N = 2048, K = 256;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> wei_scale(N);
    std::vector<bfloat16_t> dst_ck(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_fallback(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = bfloat16_t((static_cast<int>(i % 17) - 8) * 0.03125f);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 9) - 4);
    for (int n = 0; n < N; ++n)
        wei_scale[n] = 0.005f + static_cast<float>(n % 7) * 0.0005f;

    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.mem_format_a = 'n';
    params.mem_format_b = 'n';
    params.weight_cache_type = 2;
    params.num_threads = 32;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    std::string mode;
    {
        mt::GemmModeCaptureGuard mode_guard;
        ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N,
                          K, 1.0f, src.data(), K, weight.data(), K,
                          /*bias=*/nullptr, 0.0f, dst_ck.data(), N,
                          /*is_weights_const=*/true, batch_params, params),
                status_t::success);
        const char *captured
                = zendnnl::lowoha::matmul::test_api::
                          s_last_group_matmul_direct_gemm_mode.load(
                                  std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        mode = captured;
    }
    EXPECT_NE(mode.find("custom"), std::string::npos)
            << "route-enabled scalar INT8 call did not execute CK; mode="
            << mode;

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, dst_fallback.data(), N,
                      /*is_weights_const=*/true, batch_params, params),
            status_t::success);

    for (size_t i = 0; i < dst_ck.size(); ++i) {
        const float actual = static_cast<float>(dst_ck[i]);
        const float expected = static_cast<float>(dst_fallback[i]);
        EXPECT_NEAR(actual, expected, 0.05f + 0.05f * std::fabs(expected))
                << "route-enabled INT8 output differs at i=" << i;
    }
}

TEST(CkDispatchRouting, ScalarStaticU8S8PerTensorMatchesFallbackAndRunsCk) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int M = 5, N = 256, K = 64;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> bias(N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(101 + (i * 7) % 43);
    for (int n = 0; n < N; ++n) {
        bias[n] = static_cast<float>((n % 7) - 3) * 0.0625f;
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * K + k]
                    = static_cast<int8_t>((n + k * 3) % 9 - 4);
    }

    float src_scale = 0.015625f;
    int32_t src_zp = 121;
    float wei_scale = 0.03125f;
    float dst_scale = 2.5f;
    int32_t dst_zp = 113;
    matmul_params base;
    base.dtypes.src = data_type_t::u8;
    base.dtypes.wei = data_type_t::s8;
    base.dtypes.bias = data_type_t::f32;
    base.dtypes.compute = data_type_t::none;
    base.mem_format_a = 'n';
    base.mem_format_b = 'n';
    base.weight_cache_type = 1;
    base.num_threads = 4;
    base.quant_params.src_scale.buff = &src_scale;
    base.quant_params.src_scale.dt = data_type_t::f32;
    base.quant_params.src_scale.dims = {1, 1};
    base.quant_params.src_zp.buff = &src_zp;
    base.quant_params.src_zp.dt = data_type_t::s32;
    base.quant_params.src_zp.dims = {1, 1};
    base.quant_params.wei_scale.buff = &wei_scale;
    base.quant_params.wei_scale.dt = data_type_t::f32;
    base.quant_params.wei_scale.dims = {1, 1};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    const auto run_case = [&](auto zero, data_type_t dst_dt) {
        using dst_t = decltype(zero);
        matmul_params params = base;
        params.dtypes.dst = dst_dt;
        if (dst_dt == data_type_t::u8) {
            params.quant_params.dst_scale.buff = &dst_scale;
            params.quant_params.dst_scale.dt = data_type_t::f32;
            params.quant_params.dst_scale.dims
                    = base.quant_params.src_scale.dims;
            params.quant_params.dst_zp.buff = &dst_zp;
            params.quant_params.dst_zp.dt = data_type_t::s32;
            params.quant_params.dst_zp.dims = base.quant_params.src_scale.dims;
        }
        std::vector<dst_t> dst_ck(static_cast<size_t>(M) * N, zero);
        std::vector<dst_t> dst_fallback(static_cast<size_t>(M) * N, zero);

        guard.config.set_custom_kernel_route(true);
        std::string mode;
        {
            mt::GemmModeCaptureGuard mode_guard;
            const status_t ck_status = zendnnl::lowoha::matmul::matmul_direct(
                    'r', false, true, M, N, K, 1.0f, src.data(), K,
                    weight.data(), K, bias.data(), 0.0f, dst_ck.data(), N, true,
                    batch_params, params);
            ASSERT_EQ(ck_status, status_t::success);
            const char *captured
                    = zendnnl::lowoha::matmul::test_api::
                              s_last_group_matmul_direct_gemm_mode.load(
                                      std::memory_order_relaxed);
            ASSERT_NE(captured, nullptr);
            mode = captured;
        }
        EXPECT_NE(mode.find("custom"), std::string::npos) << mode;

        ::reset_grp_matmul_caches();
        guard.config.set_custom_kernel_route(false);
        ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N,
                          K, 1.0f, src.data(), K, weight.data(), K, bias.data(),
                          0.0f, dst_fallback.data(), N, true, batch_params,
                          params),
                status_t::success);

        for (size_t i = 0; i < dst_ck.size(); ++i) {
            if constexpr (std::is_same_v<dst_t, uint8_t>) {
                EXPECT_EQ(dst_ck[i], dst_fallback[i]) << "i=" << i;
            } else {
                const float actual = static_cast<float>(dst_ck[i]);
                const float expected = static_cast<float>(dst_fallback[i]);
                EXPECT_NEAR(
                        actual, expected, 0.01f + 0.01f * std::fabs(expected))
                        << "i=" << i;
            }
        }
        ::reset_grp_matmul_caches();
    };

    for (const std::vector<int64_t> &dims :
            std::vector<std::vector<int64_t>> {{}, {1}, {1, 1}}) {
        base.quant_params.src_scale.dims = dims;
        base.quant_params.src_zp.dims = dims;
        base.quant_params.wei_scale.dims = dims;
        run_case(bfloat16_t(0.0f), data_type_t::bf16);
        run_case(0.0f, data_type_t::f32);
        run_case(uint8_t {0}, data_type_t::u8);
    }

    // Static U8xS8 callers may quantize symmetric S8 weights per output
    // channel while keeping the U8 source scale/zp per tensor. Exercise the
    // same scalar route with a genuinely varying `{N}` scale so an incorrect
    // scalar broadcast cannot pass by coincidence.
    std::vector<float> wei_scales(N);
    for (int n = 0; n < N; ++n)
        wei_scales[n] = 0.015625f * static_cast<float>(1 + (n % 4));
    base.quant_params.src_scale.dims = {1};
    base.quant_params.src_zp.dims = {1};
    base.quant_params.wei_scale.buff = wei_scales.data();
    for (const std::vector<int64_t> &wei_dims :
            std::vector<std::vector<int64_t>> {{N}, {1, N}}) {
        base.quant_params.wei_scale.dims = wei_dims;
        run_case(bfloat16_t(0.0f), data_type_t::bf16);
        run_case(0.0f, data_type_t::f32);
        run_case(uint8_t {0}, data_type_t::u8);
    }

    std::vector<float> binary_mul(static_cast<size_t>(M) * N);
    std::vector<float> binary_add(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < binary_mul.size(); ++i) {
        binary_mul[i] = 0.5f + static_cast<float>(i % 5) * 0.125f;
        binary_add[i]
                = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.03125f;
    }
    matmul_post_op mul;
    mul.po_type = post_op_type_t::binary_mul;
    mul.buff = binary_mul.data();
    mul.dtype = data_type_t::f32;
    mul.dims = {M, N};
    mul.leading_dim = -1;
    matmul_post_op add;
    add.po_type = post_op_type_t::binary_add;
    add.buff = binary_add.data();
    add.dtype = data_type_t::f32;
    add.dims = {M, N};
    add.leading_dim = -1;
    base.postop_ = {mul, add};
    EXPECT_TRUE(zendnnl::lowoha::matmul::int8_ck_binary_postops_supported(
            base, M, N))
            << "the shared predicate must interpret leading_dim=-1 as N "
               "before entry-point normalization";
    EXPECT_TRUE(zendnnl::lowoha::matmul::int8_ck_postops_supported(base, M, N));
    run_case(bfloat16_t(0.0f), data_type_t::bf16);
    run_case(0.0f, data_type_t::f32);
    run_case(uint8_t {0}, data_type_t::u8);

    matmul_params unsupported_postop = base;
    unsupported_postop.dtypes.dst = data_type_t::f32;
    for (const std::vector<int64_t> &bad_dims :
            std::vector<std::vector<int64_t>> {{1, N}, {M * N}}) {
        unsupported_postop.postop_[1].dims = bad_dims;
        EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N,
                K, 4, 1, false, true, 1.0f, 0.0f, bias.data(), K, true,
                unsupported_postop));
    }
    unsupported_postop.postop_[1].dims = {M, N};
    unsupported_postop.postop_[1].leading_dim = N - 1;
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, bias.data(), K, true,
            unsupported_postop));

    for (post_op_type_t unary :
            {post_op_type_t::relu, post_op_type_t::sigmoid}) {
        matmul_post_op po;
        po.po_type = unary;
        base.postop_ = {po};
        run_case(bfloat16_t(0.0f), data_type_t::bf16);
        run_case(0.0f, data_type_t::f32);
        run_case(uint8_t {0}, data_type_t::u8);
    }
}

TEST(CkDispatchRouting, ScalarStaticU8S8PromptUsesGroupedAoclWarmup) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::AlgoEnvGuard auto_group_algo(0);
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int M = 256, N = 256, K = 64;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K, uint8_t {123});
    std::vector<int8_t> weight(static_cast<size_t>(N) * K, int8_t {1});
    std::vector<bfloat16_t> dst_route_on(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_route_off(static_cast<size_t>(M) * N);
    float src_scale = 0.015625f, wei_scale = 0.03125f;
    int32_t src_zp = 121;
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.num_threads = 4;
    params.weight_cache_type = 1;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, 1};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K, 4,
            1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', 2048, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));

    constexpr int decode_m = 5;
    EXPECT_TRUE(zendnnl::lowoha::matmul::custom_kernel_routable('r', decode_m,
            N, K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', decode_m,
            N, K, 1, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params))
            << "one-op grouped AUTO selects ALGO 1 on a one-thread call";
    {
        mt::AlgoEnvGuard force_group_algo1(1);
        EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r',
                decode_m, N, K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true,
                params));
    }
    {
        mt::AutoDecodeAlgoOverride force_decode_algo1(1);
        EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r',
                decode_m, N, K, 4, 1, false, true, 1.0f, 0.0f, nullptr, K, true,
                params));
    }

    const status_t route_on = zendnnl::lowoha::matmul::matmul_direct('r', false,
            true, M, N, K, 1.0f, src.data(), K, weight.data(), K, nullptr, 0.0f,
            dst_route_on.data(), N, true, batch_params, params);
    ASSERT_EQ(route_on, status_t::success);

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K, nullptr, 0.0f,
                      dst_route_off.data(), N, true, batch_params, params),
            status_t::success);
    EXPECT_EQ(std::memcmp(dst_route_on.data(), dst_route_off.data(),
                      dst_route_on.size() * sizeof(bfloat16_t)),
            0);
}

TEST(CkDispatchRouting,
        ScalarStaticU8S8Nr64HandlesPaddedStridesAndMultipleMrChunks) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::CustomKernelNROverride nr64(64);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int M = 13, N = 512, K = 64;
    constexpr int lda = K + 7, ldc = N + 11;
    std::vector<uint8_t> src(static_cast<size_t>(M) * lda, uint8_t {0});
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    const bfloat16_t canary(-17.0f);
    std::vector<bfloat16_t> dst_ck(static_cast<size_t>(M) * ldc, canary);
    std::vector<bfloat16_t> dst_ref(static_cast<size_t>(M) * ldc, canary);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            src[static_cast<size_t>(m) * lda + k]
                    = static_cast<uint8_t>(101 + (m * 7 + k * 3) % 43);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 9) - 4);

    float src_scale = 0.015625f, wei_scale = 0.03125f;
    int32_t src_zp = 121;
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.num_threads = 4;
    params.weight_cache_type = 1;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1};
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    const status_t ck_status = zendnnl::lowoha::matmul::matmul_direct('r',
            false, true, M, N, K, 1.0f, src.data(), lda, weight.data(), K,
            nullptr, 0.0f, dst_ck.data(), ldc, true, batch_params, params);
    ASSERT_EQ(ck_status, status_t::success);

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), lda, weight.data(), K, nullptr, 0.0f,
                      dst_ref.data(), ldc, true, batch_params, params),
            status_t::success);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            EXPECT_NEAR(static_cast<float>(
                                dst_ck[static_cast<size_t>(m) * ldc + n]),
                    static_cast<float>(
                            dst_ref[static_cast<size_t>(m) * ldc + n]),
                    0.01f);
        }
        for (int n = N; n < ldc; ++n) {
            EXPECT_FLOAT_EQ(static_cast<float>(
                                    dst_ck[static_cast<size_t>(m) * ldc + n]),
                    static_cast<float>(canary));
        }
    }
}

TEST(CkDispatchRouting,
        ScalarStaticU8S8UnsupportedGranularityFallsBackNumerically) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int M = 5, N = 256, K = 64;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> wei_scale(N);
    std::vector<bfloat16_t> dst_route_on(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_route_off(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(103 + (i * 5) % 37);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 7) - 3);
    for (size_t n = 0; n < wei_scale.size(); ++n)
        wei_scale[n] = 0.02f + static_cast<float>(n % 5) * 0.001f;

    float src_scale = 0.015625f;
    int32_t src_zp = 119;
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.mem_format_a = 'n';
    params.mem_format_b = 'n';
    params.weight_cache_type = 1;
    params.num_threads = 4;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    // The element count is N so AOCL's full-N fallback can consume it, but
    // `{N,1}` is not one of CK's explicit per-channel forms (`{N}`/`{1,N}`).
    params.quant_params.wei_scale.dims = {N, 1};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            4, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    const status_t fallback_status = zendnnl::lowoha::matmul::matmul_direct('r',
            false, true, M, N, K, 1.0f, src.data(), K, weight.data(), K,
            nullptr, 0.0f, dst_route_on.data(), N, true, batch_params, params);
    ASSERT_EQ(fallback_status, status_t::success);

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K, nullptr, 0.0f,
                      dst_route_off.data(), N, true, batch_params, params),
            status_t::success);
    EXPECT_EQ(std::memcmp(dst_route_on.data(), dst_route_off.data(),
                      dst_route_on.size() * sizeof(bfloat16_t)),
            0);
}

TEST(CkDispatchRouting,
        GroupedStaticU8S8MissingMetadataFailsClosedForPrepackedCk) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    constexpr int M = 5, N = 256, K = 64;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K, uint8_t {127});
    std::vector<int8_t> raw_weight(static_cast<size_t>(N) * K, int8_t {1});
    const int8_t *packed = nullptr;
    ASSERT_EQ(ck::get_or_pack_weight_int8(raw_weight.data(), K, N, K, 32,
                      /*transB=*/true, /*interleave_split_halves=*/false,
                      &packed, nullptr, /*disable_cache=*/true),
            status_t::success);
    ASSERT_NE(packed, nullptr);
    struct PackedGuard {
        const int8_t *ptr;
        ~PackedGuard() { ck::free_owned_packed_weight_int8(ptr); }
    } packed_guard {packed};

    const bfloat16_t canary(7.0f);
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N, canary);
    matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.mem_format_b = 'r';
    params.lowoha_algo = zendnnl::common::matmul_algo_t::moe_custom_kernel;
    params.num_threads = 4;

    mt::AlgoEnvGuard algo_guard(0);
    mt::CustomKernelOverride ck_master(true);
    mt::CustomKernelInt8Override ck_int8(true);
    const status_t result = group_matmul_direct(std::vector<char> {'r'},
            std::vector<bool> {false}, std::vector<bool> {true},
            std::vector<int> {M}, std::vector<int> {N}, std::vector<int> {K},
            std::vector<float> {1.0f}, std::vector<const void *> {src.data()},
            std::vector<int> {K}, std::vector<const void *> {packed},
            std::vector<int> {K}, std::vector<const void *> {nullptr},
            std::vector<float> {0.0f}, std::vector<void *> {dst.data()},
            std::vector<int> {N}, std::vector<bool> {true},
            std::vector<matmul_params> {params});
    EXPECT_EQ(result, status_t::failure);
    for (const auto value : dst)
        EXPECT_FLOAT_EQ(static_cast<float>(value), static_cast<float>(canary));
}

TEST(CkDispatchRouting, GroupedStaticU8S8MixedScaleStorageAndPostopFallback) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard config_guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    config_guard.config.set_weight_cache(1);
    omp_set_dynamic(0);

    constexpr int E = 2, N = 256, K = 64;
    const std::vector<int> Ms {5, 3};
    std::vector<std::vector<uint8_t>> src(E);
    std::vector<std::vector<int8_t>> weight(E);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(Ms[e]) * K);
        weight[e].resize(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = static_cast<uint8_t>(97 + (i * 7 + e * 11) % 53);
        for (size_t i = 0; i < weight[e].size(); ++i) {
            weight[e][i]
                    = static_cast<int8_t>(static_cast<int>((i + e) % 9) - 4);
        }
    }
    float src_scale_f32 = 0.015625f, wei_scale_f32 = 0.03125f;
    bfloat16_t src_scale_bf16(0.02f), wei_scale_bf16(0.0275f);
    int32_t src_zp[E] = {121, 109};

    std::vector<matmul_params> params(E);
    for (int e = 0; e < E; ++e) {
        auto &p = params[e];
        p.dtypes.src = data_type_t::u8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.bias = data_type_t::none;
        p.num_threads = 4;
        p.weight_cache_type = 1;
        p.quant_params.src_scale.buff = (e == 0)
                ? static_cast<const void *>(&src_scale_f32)
                : static_cast<const void *>(&src_scale_bf16);
        p.quant_params.src_scale.dt
                = (e == 0) ? data_type_t::f32 : data_type_t::bf16;
        p.quant_params.src_scale.dims = {1, 1};
        p.quant_params.src_zp.buff = &src_zp[e];
        p.quant_params.src_zp.dt = data_type_t::s32;
        p.quant_params.src_zp.dims = {1, 1};
        p.quant_params.wei_scale.buff = (e == 0)
                ? static_cast<const void *>(&wei_scale_f32)
                : static_cast<const void *>(&wei_scale_bf16);
        p.quant_params.wei_scale.dt
                = (e == 0) ? data_type_t::f32 : data_type_t::bf16;
        p.quant_params.wei_scale.dims = {1, 1};
    }

    const std::vector<char> layout(E, 'r');
    const std::vector<bool> transA(E, false), transB(E, true);
    const std::vector<int> Ns(E, N), Ks(E, K), lda(E, K), ldb(E, K);
    std::vector<int> ldc(E, N);
    const std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    std::vector<const void *> src_ptr(E), weight_ptr(E), bias(E, nullptr);
    std::vector<bool> is_const(E, true);
    for (int e = 0; e < E; ++e) {
        src_ptr[e] = src[e].data();
        weight_ptr[e] = weight[e].data();
    }

    const auto run = [&](int algo, const std::vector<matmul_params> &run_params,
                             std::vector<std::vector<bfloat16_t>> &storage,
                             const mt::grp_matmul_gated_act_params *gated_act
                             = nullptr) {
        mt::AlgoEnvGuard algo_guard(algo);
        std::vector<void *> dst(E);
        for (int e = 0; e < E; ++e) {
            storage[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
            dst[e] = storage[e].data();
        }
        const status_t result = group_matmul_direct(layout, transA, transB, Ms,
                Ns, Ks, alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst,
                ldc, is_const, run_params, /*moe_postop=*/nullptr, gated_act);
        return result;
    };

    std::vector<std::vector<bfloat16_t>> ck_out(E), ref_out(E);
    std::string initial_mode;
    {
        mt::GemmModeCaptureGuard mode_guard;
        ASSERT_EQ(run(3, params, ck_out), status_t::success);
        const char *captured
                = zendnnl::lowoha::matmul::test_api::
                          s_last_group_matmul_direct_gemm_mode.load(
                                  std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        initial_mode = captured;
    }
    EXPECT_NE(initial_mode.find("custom"), std::string::npos) << initial_mode;
    ::reset_grp_matmul_caches();
    ASSERT_EQ(run(1, params, ref_out), status_t::success);
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < ck_out[e].size(); ++i) {
            const float actual = static_cast<float>(ck_out[e][i]);
            const float expected = static_cast<float>(ref_out[e][i]);
            EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected));
        }
    }

    // Dense binary U8xS8 epilogue: leading_dim=-1 means logical N stride.
    // The exact [M,N] contract is admitted by CK and rejected by ALGO 3's
    // tiled AOCL fallback, which cannot re-anchor the full-N buffers.
    std::vector<std::vector<float>> binary_mul(E), binary_add(E);
    for (int e = 0; e < E; ++e) {
        binary_mul[e].resize(static_cast<size_t>(Ms[e]) * N);
        binary_add[e].resize(static_cast<size_t>(Ms[e]) * N);
        for (size_t i = 0; i < binary_mul[e].size(); ++i) {
            binary_mul[e][i] = 0.5f + static_cast<float>((i + e) % 5) * 0.125f;
            binary_add[e][i]
                    = static_cast<float>(static_cast<int>((i + 3 * e) % 9) - 4)
                    * 0.03125f;
        }
        matmul_post_op mul;
        mul.po_type = post_op_type_t::binary_mul;
        mul.buff = binary_mul[e].data();
        mul.dtype = data_type_t::f32;
        mul.dims = {Ms[e], N};
        mul.leading_dim = -1;
        matmul_post_op add;
        add.po_type = post_op_type_t::binary_add;
        add.buff = binary_add[e].data();
        add.dtype = data_type_t::f32;
        add.dims = {Ms[e], N};
        add.leading_dim = -1;
        params[e].postop_ = {mul, add};
    }

    std::vector<std::vector<bfloat16_t>> binary_ck(E), binary_ref(E);
    ::reset_grp_matmul_caches();
    std::string binary_mode;
    {
        mt::GemmModeCaptureGuard mode_guard;
        ASSERT_EQ(run(3, params, binary_ck), status_t::success);
        const char *captured
                = zendnnl::lowoha::matmul::test_api::
                          s_last_group_matmul_direct_gemm_mode.load(
                                  std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        binary_mode = captured;
    }
    EXPECT_NE(binary_mode.find("custom"), std::string::npos) << binary_mode;
    ::reset_grp_matmul_caches();
    ASSERT_EQ(run(1, params, binary_ref), status_t::success);
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < binary_ck[e].size(); ++i) {
            const float actual = static_cast<float>(binary_ck[e][i]);
            const float expected = static_cast<float>(binary_ref[e][i]);
            EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected));
        }
    }

    // If CK is disabled after ALGO 3 was selected, execute the dense binary
    // chain once over full N. Per-tile AOCL would incorrectly reuse columns
    // [0,n_tile) of both buffers on every tile.
    std::vector<std::vector<bfloat16_t>> ck_off_out(E);
    std::string ck_off_mode;
    ::reset_grp_matmul_caches();
    {
        mt::CustomKernelInt8Override ck_int8_off(false);
        mt::GemmModeCaptureGuard mode_guard;
        ASSERT_EQ(run(3, params, ck_off_out), status_t::success);
        const char *captured
                = zendnnl::lowoha::matmul::test_api::
                          s_last_group_matmul_direct_gemm_mode.load(
                                  std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        ck_off_mode = captured;
    }
    EXPECT_EQ(ck_off_mode, "flat_n_tile_sequential");
    for (int e = 0; e < E; ++e) {
        EXPECT_EQ(std::memcmp(ck_off_out[e].data(), binary_ref[e].data(),
                          ck_off_out[e].size() * sizeof(bfloat16_t)),
                0);
    }

    // A per-call CK refusal (non-constant weights) has the same full-N
    // fallback requirement even while the process-wide CK knob stays on.
    std::vector<std::vector<bfloat16_t>> ck_refused_out(E);
    std::string ck_refused_mode;
    std::fill(is_const.begin(), is_const.end(), false);
    ::reset_grp_matmul_caches();
    {
        mt::GemmModeCaptureGuard mode_guard;
        ASSERT_EQ(run(3, params, ck_refused_out), status_t::success);
        const char *captured
                = zendnnl::lowoha::matmul::test_api::
                          s_last_group_matmul_direct_gemm_mode.load(
                                  std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        ck_refused_mode = captured;
    }
    std::fill(is_const.begin(), is_const.end(), true);
    EXPECT_EQ(ck_refused_mode, "flat_n_tile_sequential");
    for (int e = 0; e < E; ++e) {
        EXPECT_EQ(std::memcmp(ck_refused_out[e].data(), binary_ref[e].data(),
                          ck_refused_out[e].size() * sizeof(bfloat16_t)),
                0);
    }

    // Gated activation and a static U8xS8 post-op cannot both be fused by the
    // current epilogue. The selector must reject ALGO 3 rather than silently
    // dropping either the binary or unary operation.
    std::vector<matmul_params> unary_params = params;
    for (auto &p : unary_params) {
        matmul_post_op relu;
        relu.po_type = post_op_type_t::relu;
        p.postop_ = {relu};
    }
    {
        mt::AlgoEnvGuard algo_guard(3);
        EXPECT_TRUE(zendnnl::lowoha::matmul::grp_matmul_custom_kernel_effective(
                params, Ms, Ns, mt::grp_matmul_gated_act_t::none));
        EXPECT_TRUE(zendnnl::lowoha::matmul::grp_matmul_custom_kernel_effective(
                unary_params, Ms, Ns, mt::grp_matmul_gated_act_t::none));
        EXPECT_FALSE(
                zendnnl::lowoha::matmul::grp_matmul_custom_kernel_effective(
                        params, Ms, Ns,
                        mt::grp_matmul_gated_act_t::swiglu_oai_mul));
        EXPECT_FALSE(
                zendnnl::lowoha::matmul::grp_matmul_custom_kernel_effective(
                        unary_params, Ms, Ns,
                        mt::grp_matmul_gated_act_t::swiglu_oai_mul));
        EXPECT_EQ(zendnnl::lowoha::matmul::select_grp_matmul_algo(layout, Ms,
                          Ns, Ks, params, /*num_threads=*/4, nullptr,
                          mt::grp_matmul_gated_act_t::swiglu_oai_mul),
                1);
        EXPECT_EQ(zendnnl::lowoha::matmul::select_grp_matmul_algo(layout, Ms,
                          Ns, Ks, unary_params, /*num_threads=*/4, nullptr,
                          mt::grp_matmul_gated_act_t::swiglu_oai_mul),
                1);
    }

    // Execution-level proof: a wide destination falls back to full-N ALGO 1
    // without dropping either post-op form. A tight destination cannot be
    // served by that fallback, so it must fail before writing instead of
    // entering gated CK and silently dropping the post-op.
    mt::grp_matmul_gated_act_params gated_act {};
    gated_act.act = mt::grp_matmul_gated_act_t::swiglu_oai_mul;
    const auto check_gated_fallback
            = [&](const std::vector<matmul_params> &run_params,
                      const char *label) {
        SCOPED_TRACE(label);
        std::vector<std::vector<bfloat16_t>> forced_three(E), forced_one(E);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(run(3, run_params, forced_three, &gated_act),
                status_t::success);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(
                run(1, run_params, forced_one, &gated_act), status_t::success);
        for (int e = 0; e < E; ++e) {
            EXPECT_EQ(std::memcmp(forced_three[e].data(), forced_one[e].data(),
                              forced_three[e].size() * sizeof(bfloat16_t)),
                    0);
        }

        ldc.assign(E, N / 2);
        std::vector<std::vector<bfloat16_t>> tight_out(E);
        ::reset_grp_matmul_caches();
        EXPECT_EQ(run(3, run_params, tight_out, &gated_act), status_t::failure);
        for (const auto &expert_out : tight_out) {
            for (const bfloat16_t value : expert_out) {
                EXPECT_FLOAT_EQ(static_cast<float>(value), 0.0f);
            }
        }
        ldc.assign(E, N);
    };
    check_gated_fallback(params, "binary_mul_add");
    check_gated_fallback(unary_params, "relu");

    matmul_post_op unsupported_postop;
    unsupported_postop.po_type = post_op_type_t::gelu_tanh;
    for (auto &p : params)
        p.postop_ = {unsupported_postop};
    ::reset_grp_matmul_caches();
    ASSERT_EQ(run(3, params, ck_out), status_t::success);
    zendnnl::lowoha::matmul::group_matmul_prepack::custom_kernel::PackProbeStats
            post_op_probe;
    ASSERT_EQ(zendnnl::lowoha::matmul::group_matmul_prepack::custom_kernel::
                      warm_pack_all_custom_kernel_experts(weight_ptr, Ks, Ns,
                              ldb, transB, is_const, E, post_op_probe,
                              /*interleave_split_halves=*/false,
                              zendnnl::lowoha::matmul::group_matmul_prepack::
                                      custom_kernel::WarmDtypeFamily::kINT8),
            status_t::success);
    EXPECT_EQ(post_op_probe.cache_hits, 0);
    EXPECT_EQ(post_op_probe.cache_misses, E)
            << "post-op fallback must not pre-populate an unused CK arena";
    ::reset_grp_matmul_caches();
    ASSERT_EQ(run(1, params, ref_out), status_t::success);
    for (int e = 0; e < E; ++e) {
        EXPECT_EQ(std::memcmp(ck_out[e].data(), ref_out[e].data(),
                          ck_out[e].size() * sizeof(bfloat16_t)),
                0);
    }

    for (auto &p : params)
        p.weight_cache_type = 2;
    config_guard.config.set_weight_cache(2);
    ::reset_grp_matmul_caches();
    ASSERT_EQ(run(0, params, ck_out), status_t::success);
    EXPECT_EQ(config_guard.config.get_weight_cache(), 1)
            << "post-op-ineligible CK must not license AUTO mixed in-place";
    ::reset_grp_matmul_caches();
}

TEST(CkDispatchRouting, ScalarInt8DstMetadataFallsBackNumerically) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    constexpr int M = 8, N = 2048, K = 256;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> wei_scale(N, 0.01f);
    std::vector<bfloat16_t> dst_route_on(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_route_off(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = bfloat16_t((static_cast<int>(i % 13) - 6) * 0.03125f);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 7) - 3);

    float dst_scale = 0.5f;
    int32_t dst_zp = 3;
    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.mem_format_a = 'n';
    params.mem_format_b = 'n';
    params.weight_cache_type = 2;
    params.num_threads = 32;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};
    params.quant_params.dst_scale.buff = &dst_scale;
    params.quant_params.dst_scale.dt = data_type_t::f32;
    params.quant_params.dst_scale.dims = {1, 1};
    params.quant_params.dst_zp.buff = &dst_zp;
    params.quant_params.dst_zp.dt = data_type_t::s32;
    params.quant_params.dst_zp.dims = {1, 1};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, dst_route_on.data(), N,
                      /*is_weights_const=*/true, batch_params, params),
            status_t::success);

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, dst_route_off.data(), N,
                      /*is_weights_const=*/true, batch_params, params),
            status_t::success);
    for (size_t i = 0; i < dst_route_on.size(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(dst_route_on[i]),
                static_cast<float>(dst_route_off[i]))
                << "destination-metadata fallback differs at i=" << i;
    }
}

TEST(CkDispatchRouting, ScalarSignedInt8SrcZpFallsBackWithGroupDqOff) {
    CK_SKIP_IF_NO_INT8_ISA();
    ::reset_grp_matmul_caches();

    ScalarRouteConfigGuard guard;
    mt::CustomKernelOverride group_ck_on(true);
    mt::CustomKernelInt8Override int8_ck_on(true);
    mt::NTileStrategyOverride rounds(2);
    mt::GroupDqOverride group_dq_off(0);
    guard.config.set_custom_kernel_route(true);
    guard.config.set_algo(static_cast<int32_t>(
            zendnnl::common::matmul_algo_t::aocl_dlp_blocked));
    guard.config.set_weight_cache(2);
    omp_set_dynamic(0);

    constexpr int M = 1, N = 2048, K = 256;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> wei_scale(N, 0.01f);
    std::vector<bfloat16_t> dst_route_on(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_route_off(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = bfloat16_t((static_cast<int>(i % 11) - 5) * 0.03125f);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 7) - 3);

    int32_t src_zp = 11;
    matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.mem_format_a = 'n';
    params.mem_format_b = 'n';
    params.weight_cache_type = 2;
    params.num_threads = 32;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};
    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    EXPECT_FALSE(zendnnl::lowoha::matmul::custom_kernel_routable('r', M, N, K,
            32, 1, false, true, 1.0f, 0.0f, nullptr, K, true, params));
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, dst_route_on.data(), N,
                      /*is_weights_const=*/true, batch_params, params),
            status_t::success);

    ::reset_grp_matmul_caches();
    guard.config.set_custom_kernel_route(false);
    ASSERT_EQ(zendnnl::lowoha::matmul::matmul_direct('r', false, true, M, N, K,
                      1.0f, src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, dst_route_off.data(), N,
                      /*is_weights_const=*/true, batch_params, params),
            status_t::success);
    for (size_t i = 0; i < dst_route_on.size(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(dst_route_on[i]),
                static_cast<float>(dst_route_off[i]))
                << "signed source-zero-point fallback differs at i=" << i;
    }
}

// ──────────────────────────────────────────────────────────────────
// `is_weights_const = false` → CK refuses → DLP path runs.  The
// output must match an equivalent call with ALGO=1 forced (which
// always uses DLP).
//
// Uses FOUR experts AND pins `params.num_threads = kNumOps` so the
// planner stays on the ntile path (where the BF16 custom kernel's
// gate actually runs).  Without the thread pin, `num_ops <
// num_threads` on a typical CI host would trip the auto-mirror
// Rule 2 gate (`num_ops <= 8` → Sequential — see
// `auto_select_would_pick_algo1` in
// `group_matmul/n_tile/group_matmul_n_tile.cpp`), bypassing the custom
// kernel entirely.  In that case the test would silently pass via
// the Sequential → AOCL path even if the CK refusal logic were
// broken: both ALGO=3 (forced, with auto-mirror Sequential) and
// ALGO=1 (forced) would run AOCL DLP and produce identical output
// without ever consulting the `is_weights_const = false` refusal
// path.  The thread pin makes `num_ops == num_threads` → Rule 1
// fires inside the auto-mirror → ntile path runs → CK gate is
// actually exercised → refusal forces the DLP fallback we want
// to compare against forced ALGO=1.
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, NonConstWeightsFallsBackToDlp) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();

    constexpr int kNumOps = 4;
    constexpr int M = 16, K = 256, N = 512;

    // Per-expert BF16 src / weight buffers (uniform shape).
    std::vector<std::vector<bfloat16_t>> src_bufs(kNumOps);
    std::vector<std::vector<bfloat16_t>> wei_bufs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        src_bufs[e].assign(static_cast<size_t>(M) * K, bfloat16_t(0.0f));
        wei_bufs[e].assign(static_cast<size_t>(K) * N, bfloat16_t(0.0f));
        mt::fill_src(src_bufs[e], /*e=*/e);
        mt::fill_wei1(wei_bufs[e], /*e=*/e);
    }
    // Per-expert dst buffers, two passes (CK-fallback vs forced ALGO 1).
    std::vector<std::vector<bfloat16_t>> dst_ck(kNumOps);
    std::vector<std::vector<bfloat16_t>> dst_dlp(kNumOps);
    for (int e = 0; e < kNumOps; ++e) {
        dst_ck[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
        dst_dlp[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
    }

    std::vector<char> layout(kNumOps, 'r');
    std::vector<bool> transA(kNumOps, false), transB(kNumOps, false);
    std::vector<int> Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
    std::vector<float> alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
    std::vector<int> lda(kNumOps, K), ldb(kNumOps, N), ldc(kNumOps, N);
    std::vector<const void *> src_ptrs(kNumOps), wei_ptrs(kNumOps);
    std::vector<const void *> bias_ptrs(kNumOps, nullptr);
    for (int e = 0; e < kNumOps; ++e) {
        src_ptrs[e] = src_bufs[e].data();
        wei_ptrs[e] = wei_bufs[e].data();
    }
    std::vector<bool> is_wc_false(kNumOps, false);

    std::vector<matmul_params> params(kNumOps);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.bias = data_type_t::none;
        // Pin the dispatcher's per-call thread team to `kNumOps` so
        // `num_ops == num_threads` → auto-select Rule 1 fires in the
        // planner's auto-mirror gate, ntile is picked (instead of
        // Sequential under the Rule 2 / Rule 3-prompt arrows), and
        // the CK gate is actually consulted on the call — letting
        // `is_weights_const = false` exercise the refusal-then-DLP
        // path this test is named for.  See the function-level
        // doc-block above for the full rationale.
        p.num_threads = kNumOps;
    }

    // Run 1: ALGO=3 forced + custom-kernel gate ON via the test-only
    // atomic override (deterministic regardless of any cached state in
    // `get_grp_matmul_custom_kernel()`) + is_weights_const=false → CK
    // refused, DLP runs.
    {
        mt::AlgoEnvGuard algo_guard(3);
        mt::CustomKernelOverride ck_guard(true);
        std::vector<void *> dst_ptrs(kNumOps);
        for (int e = 0; e < kNumOps; ++e)
            dst_ptrs[e] = dst_ck[e].data();
        ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                          src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta,
                          dst_ptrs, ldc, is_wc_false, params,
                          /*moe_postop=*/nullptr,
                          /*gated_act=*/nullptr),
                status_t::success);
    }

    ::reset_grp_matmul_caches();
    // Run 2: ALGO=1 forced (always DLP) — reference for the comparison.
    {
        mt::AlgoEnvGuard algo_guard(1);
        std::vector<void *> dst_ptrs(kNumOps);
        for (int e = 0; e < kNumOps; ++e)
            dst_ptrs[e] = dst_dlp[e].data();
        ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks, alpha,
                          src_ptrs, lda, wei_ptrs, ldb, bias_ptrs, beta,
                          dst_ptrs, ldc, is_wc_false, params,
                          /*moe_postop=*/nullptr,
                          /*gated_act=*/nullptr),
                status_t::success);
    }

    // Outputs must be bit-identical for every expert (both ran the
    // same DLP path).
    for (int e = 0; e < kNumOps; ++e) {
        for (int i = 0; i < M * N; ++i) {
            ASSERT_EQ(static_cast<float>(dst_ck[e][i]),
                    static_cast<float>(dst_dlp[e][i]))
                    << "CK-refused (fallback) output deviates from "
                       "forced-ALGO-1 "
                       "DLP output at expert="
                    << e << " i=" << i;
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// dst_dt = f16 → CK refuses → DLP path runs.  Currently DLP also
// doesn't support f16 dst on this configuration, so this test only
// asserts that the dispatcher REFUSES the call cleanly (returns
// status_t::failure) rather than silently producing garbage.
//
// If a future ALGO 1 / DLP code path accepts f16 dst, this test
// should be tightened to compare against that reference (mirroring
// the is_weights_const test above).
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, F16DstRefusedAtCkLayer) {
    // resolve_variant must reject f16 dst.
    EXPECT_EQ(ck::resolve_variant(
                      data_type_t::bf16, data_type_t::bf16, data_type_t::f16),
            ck::KernelVariant::kUnsupported);

    // prepare_for_call must also refuse on f16 dst, regardless of host ISA.
    if (ck::dispatch_supported()) {
        ck_test::PrepCallCase c {};
        c.dst_dt = data_type_t::f16;
        c.label = "f16_dst_refusal";
        ck_test::PrepCallStorage storage;
        ck::CallContext kctx;
        EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure);
        EXPECT_FALSE(kctx.enabled);
        EXPECT_EQ(kctx.variant, ck::KernelVariant::kUnsupported);
    }
}

// ──────────────────────────────────────────────────────────────────
// resolve_variant + prepare_for_call agree on rejected dtype tuples
// when probed through the BF16-only (3-arg / `dynamic_quant=false`)
// surface.  The s8-bearing tuples below are not served by the kernel
// in the static-quant regime; `resolve_variant` returns
// `kUnsupported` and `prepare_for_call` must refuse for the same
// inputs.  The DQ-INT8 family (which DOES accept `(bf16, s8, bf16)`
// when `dynamic_quant=true` and `compute_dtype ∈ {s8, u8}`) is
// covered by the int8 truth-table tests in test_resolve_variant.cpp
// and the prepare-for-call int8 tests in test_prepare_for_call.cpp;
// the rows here probe ONLY the static-quant fallback.  Locking the
// symmetry means a later change that adds new dtype rows updates
// both layers together.
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, ResolveAndPrepareAgreeOnRejectedTuples) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "ISA gate would refuse anyway";
    }
    struct Row {
        data_type_t src, wei, dst;
    };
    for (auto r : {Row {data_type_t::bf16, data_type_t::s8, data_type_t::bf16},
                 Row {data_type_t::bf16, data_type_t::s8, data_type_t::f32},
                 Row {data_type_t::s8, data_type_t::s8, data_type_t::bf16},
                 Row {data_type_t::s8, data_type_t::s8, data_type_t::f32}}) {
        // resolve_variant rejects.
        EXPECT_EQ(ck::resolve_variant(r.src, r.wei, r.dst),
                ck::KernelVariant::kUnsupported)
                << "src=" << ck_test::dt_name(r.src)
                << " wei=" << ck_test::dt_name(r.wei)
                << " dst=" << ck_test::dt_name(r.dst)
                << " — if this row now resolves to a real variant, move it "
                   "to test_resolve_variant.cpp's positive table.";
        // prepare_for_call rejects too.
        ck_test::PrepCallCase c {};
        c.src_dt = r.src;
        c.wei_dt = r.wei;
        c.dst_dt = r.dst;
        c.label = "rejected_dtype_tuple";
        ck_test::PrepCallStorage storage;
        ck::CallContext kctx;
        EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure);
        EXPECT_FALSE(kctx.enabled);
    }
}

// ──────────────────────────────────────────────────────────────────
// C.4 — `CallContext` cross-reuse between bf16 and int8 families.
//
// `prepare_for_call` is documented to reset the `CallContext` on
// every entry.  A single instance must be reusable across a bf16
// call followed by an int8 call (a realistic pattern: a single
// caller that prepares both families during a layer transition).
// Verify that:
//   1. bf16 prepare populates kctx.variant ∈ {kBF16_*}.
//   2. int8 prepare on the SAME kctx fully replaces it (variant
//      becomes int8, kctx.enabled is true, the int8 weight cache
//      key was populated, the bf16 cache key cleared).
//   3. A return to bf16 again on the same kctx works the third
//      time (rules out one-shot state in the dispatcher).
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, CallContextReusableAcrossFamilies) {
    CK_SKIP_IF_NO_INT8_ISA();

    ck::CallContext kctx;
    ck_test::PrepCallStorage storage;

    // (1) bf16 call.
    {
        ck_test::PrepCallCase c {};
        c.label = "reuse_bf16_first";
        c.src_dt = data_type_t::bf16;
        c.wei_dt = data_type_t::bf16;
        c.dst_dt = data_type_t::bf16;
        ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
        EXPECT_TRUE(kctx.enabled);
        EXPECT_FALSE(ck::is_int8_variant(kctx.variant))
                << "bf16 call must yield a bf16 variant";
    }

    // (2) int8 call on the SAME kctx.
    {
        ck_test::PrepCallCase c {};
        c.label = "reuse_int8_after_bf16";
        c.src_dt = data_type_t::bf16;
        c.wei_dt = data_type_t::s8;
        c.dst_dt = data_type_t::bf16;
        c.dynamic_quant = true;
        c.compute_dt = data_type_t::s8;
        ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
        EXPECT_TRUE(kctx.enabled);
        EXPECT_TRUE(ck::is_int8_variant(kctx.variant))
                << "int8 call on a previously-bf16-used kctx must yield an "
                   "int8 variant (state must be fully reset on every entry)";
        EXPECT_EQ(kctx.variant, ck::KernelVariant::kS8_S8_BF16_SYM);
    }

    // (3) bf16 call again.
    {
        ck_test::PrepCallCase c {};
        c.label = "reuse_bf16_after_int8";
        c.src_dt = data_type_t::bf16;
        c.wei_dt = data_type_t::bf16;
        c.dst_dt = data_type_t::bf16;
        ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
        EXPECT_TRUE(kctx.enabled);
        EXPECT_FALSE(ck::is_int8_variant(kctx.variant))
                << "third call must return cleanly to bf16; reuse must not "
                   "be one-shot";
    }
}

// ──────────────────────────────────────────────────────────────────
// C.4 — bad-hoist routing.  When the caller passes
// `dynamic_quant=true` with `wei.dtype != s8`, the
// `reorder_quantization_wrapper` short-circuits (no-op) and the
// pre-OMP hoist loop in `flat_n_tile` leaves `hoisted[e].valid =
// false`.  Phase 1 hardening (B.1) added a defensive `do_tile`
// fall-through to the AOCL path in that case.  We verify the
// behaviour structurally at the `prepare_for_call` level: a
// `(bf16, bf16, bf16)` tuple with `dynamic_quant=true` is NOT in
// the accepted-int8 truth table, so prepare_for_call must refuse
// (the dispatcher catches the inconsistency before any hoist runs).
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, DynamicQuantWithBf16WeiRefused) {
    CK_SKIP_IF_NO_INT8_ISA();
    ck_test::PrepCallCase c {};
    c.label = "dq_with_bf16_wei_refused";
    c.src_dt = data_type_t::bf16;
    c.wei_dt = data_type_t::bf16; // wrong for dynamic_quant=true
    c.dst_dt = data_type_t::bf16;
    c.dynamic_quant = true;
    c.compute_dt = data_type_t::s8;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure)
            << "prepare_for_call must refuse dynamic_quant=true with a "
               "non-s8 weight; otherwise the int8 ukernel would consume "
               "the bf16 byte stream as if it were s8 and silently corrupt "
               "every accumulator (B.1 hardening defends a related case "
               "at the do_tile level for completeness, but the structural "
               "refusal at the gate is the primary defence).";
    EXPECT_FALSE(kctx.enabled);
}

} // namespace
