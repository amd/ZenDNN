/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/matmul_route.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/prompt_tile/prompt_tile_tunables.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "moe_test_utils.hpp"

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
namespace matmul = zendnnl::lowoha::matmul;
namespace reorder = zendnnl::lowoha::reorder;
namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace mt = moe_test_utils;

struct aligned_deleter {
    void operator()(void *ptr) const noexcept {
        if (ptr != nullptr) zendnnl_aligned_free(ptr);
    }
};
using aligned_ptr_t = std::unique_ptr<void, aligned_deleter>;

class RouteConfigGuard {
public:
    RouteConfigGuard()
        : config_(matmul_config_t::instance())
        , algo_(config_.get_algo())
        , route_(config_.get_custom_kernel_route())
        , capacity_(config_.get_lru_cache_capacity()) {
        config_.set_algo(static_cast<int32_t>(matmul_algo_t::aocl_dlp_blocked));
        config_.set_custom_kernel_route(true);
        config_.set_lru_cache_capacity(std::numeric_limits<uint32_t>::max());
    }
    ~RouteConfigGuard() {
        config_.set_lru_cache_capacity(capacity_);
        config_.set_custom_kernel_route(route_);
        config_.set_algo(algo_);
    }

private:
    matmul_config_t &config_;
    int32_t algo_;
    bool route_;
    uint32_t capacity_;
};

float logical_weight(int k, int n) {
    return 0.0025f * static_cast<float>((k * 17 + n * 7) % 19 - 9);
}

void run_route_case(bool packed_input, int M = 8, bool expect_custom = true,
        bool expect_derived_cache_hit = true, bool trans_b = true,
        int ldb_padding = 0, int weight_cache_mode = 1,
        bool direct_group = false, float alpha_value = 1.0f,
        bool probe_derived_cache = true, bool relu_postop = false,
        matmul_algo_t per_op_algo = matmul_algo_t::none,
        matmul_algo_t global_algo = matmul_algo_t::aocl_dlp_blocked) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (packed_input) GTEST_SKIP() << "AOCL-DLP is unavailable";
#endif

    constexpr int K = 64;
    constexpr int N = 2048;
    constexpr int threads = 8;
    const int ldb = (trans_b ? K : N) + ldb_padding;
    const int weight_rows = trans_b ? N : K;

    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<bfloat16_t> weight(static_cast<size_t>(weight_rows) * ldb);
    std::vector<float> bias(N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            src[static_cast<size_t>(m) * K + k] = bfloat16_t(
                    0.01f * static_cast<float>((m * 11 + k * 3) % 13 - 6));
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            weight[trans_b ? static_cast<size_t>(n) * ldb + k
                           : static_cast<size_t>(k) * ldb + n]
                    = bfloat16_t(logical_weight(k, n));
    for (int n = 0; n < N; ++n)
        bias[n] = 0.001f * static_cast<float>(n % 7 - 3);

    aligned_ptr_t packed;
    const void *route_weight = weight.data();
    if (packed_input) {
        reorder::reorder_params_t prepack;
        prepack.is_prepack = true;
        prepack.num_threads = threads;
        prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
        prepack.prepack.wei_dtype = data_type_t::bf16;
        prepack.prepack.src_dtype = data_type_t::bf16;
        prepack.prepack.K = K;
        prepack.prepack.N = N;
        prepack.prepack.ldb = ldb;
        prepack.prepack.transposed = trans_b;
        const size_t bytes = reorder::weight_prepack_size(prepack);
        ASSERT_GT(bytes, 0u);
        packed.reset(zendnnl_aligned_alloc(64, bytes));
        ASSERT_NE(packed, nullptr);
        ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
                status_t::success);
        route_weight = packed.get();
    }

    RouteConfigGuard route_config;
    matmul_config_t::instance().set_algo(static_cast<int32_t>(global_algo));
    WeightCacheGuard cache_mode(weight_cache_mode);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::NTileStrategyOverride rounds(2);
    matmul::clear_aocl_ck_weight_cache();

    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::f32;
    params.mem_format_b = packed_input ? 'r' : 'n';
    // The default matches scalar ZenTorch prepack (layout-only provenance);
    // targeted cases can publish the explicit producing backend.
    params.lowoha_algo = per_op_algo;
    params.num_threads = threads;
    if (relu_postop) {
        matmul::matmul_post_op relu;
        relu.po_type = zendnnl::ops::post_op_type_t::relu;
        params.postop_.push_back(relu);
    }
    matmul::matmul_batch_params_t batch;

    mt::GemmModeCaptureGuard mode_capture;
    if (direct_group) {
        ASSERT_EQ(matmul::group_matmul_direct(
                          /*layout=*/ {'r'}, /*transA=*/ {false},
                          /*transB=*/ {trans_b}, /*M=*/ {M}, /*N=*/ {N},
                          /*K=*/ {K},
                          /*alpha=*/ {alpha_value}, /*src=*/ {src.data()},
                          /*lda=*/ {K}, /*weight=*/ {route_weight},
                          /*ldb=*/ {ldb},
                          /*bias=*/ {bias.data()}, /*beta=*/ {0.0f},
                          /*dst=*/ {dst.data()}, /*ldc=*/ {N},
                          /*is_weights_const=*/ {true}, /*params=*/ {params},
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success);
    } else {
        ASSERT_EQ(
                matmul::matmul_direct('r', false, trans_b, M, N, K, alpha_value,
                        src.data(), K, route_weight, ldb, bias.data(), 0.0f,
                        dst.data(), N, true, batch, params),
                status_t::success);
    }
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_gemm_mode.load(
                    std::memory_order_relaxed);
    if (expect_custom) {
        ASSERT_NE(mode, nullptr);
        EXPECT_NE(std::string(mode).find("custom"), std::string::npos);
    } else if (mode != nullptr) {
        EXPECT_EQ(std::string(mode).find("custom"), std::string::npos);
    }

    if (packed_input && probe_derived_cache) {
        std::shared_ptr<void> ck_weight;
        bool hit = false;
        ASSERT_EQ(matmul::get_or_cache_aocl_ck_weight(packed.get(), K, N, ldb,
                          trans_b, threads, ck::plan_pack_nr(K, N),
                          /*interleave_split_halves=*/false, ck_weight, &hit),
                status_t::success);
        EXPECT_EQ(hit, expect_derived_cache_hit);
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float reference = bias[n];
            float product = 0.0f;
            for (int k = 0; k < K; ++k) {
                product += static_cast<float>(
                                   src[static_cast<size_t>(m) * K + k])
                        * static_cast<float>(bfloat16_t(logical_weight(k, n)));
            }
            reference += alpha_value * product;
            if (relu_postop) reference = std::max(0.0f, reference);
            const float expected = static_cast<float>(bfloat16_t(reference));
            const float actual
                    = static_cast<float>(dst[static_cast<size_t>(m) * N + n]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.02f + 1.0e-4f);
        }
    }

    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, PackedWeightUsesCachedDerivedCkPath) {
    run_route_case(true);
}

TEST(TestMatmulUnreorderRoute,
        ExplicitAoclPackedPromptOverridesGlobalOneDnnBackend) {
    // The grouped route serves the prompt either way: AOCL with the BF16
    // single-expert prompt tile off, the CK prompt tile with it on.
    for (const bool prompt_tile : {false, true}) {
        SCOPED_TRACE(prompt_tile ? "prompt tile on" : "prompt tile off");
        mt::SingleExpertPromptTileOverride knob(prompt_tile ? 1 : 0);
        run_route_case(/*packed_input=*/true, /*M=*/64,
                /*expect_custom=*/prompt_tile,
                /*expect_derived_cache_hit=*/false,
                /*trans_b=*/true, /*ldb_padding=*/0, /*weight_cache_mode=*/1,
                /*direct_group=*/false, /*alpha_value=*/1.0f,
                /*probe_derived_cache=*/false, /*relu_postop=*/false,
                /*per_op_algo=*/matmul_algo_t::aocl_dlp_blocked,
                /*global_algo=*/matmul_algo_t::onednn_blocked);
    }
}

TEST(TestMatmulUnreorderRoute, PackedWeightWc2UsesCachedDerivedCkPath) {
    run_route_case(true, /*M=*/8, /*expect_custom=*/true,
            /*expect_derived_cache_hit=*/true, /*trans_b=*/true,
            /*ldb_padding=*/0, /*weight_cache_mode=*/2);
}

TEST(TestMatmulUnreorderRoute, RawWeightUsesDirectGroupPath) {
    run_route_case(false);
}

TEST(TestMatmulUnreorderRoute, RawWc2IneligiblePromptPreservesFutureDecode) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    constexpr int K = 64, N = 1024, prompt_M = 64, decode_M = 8;
    constexpr int prompt_threads = 32, decode_threads = 8;
    std::vector<bfloat16_t> weight(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * K + k]
                    = bfloat16_t(logical_weight(k, n));
    const auto pristine_weight = weight;

    auto make_src = [&](int rows) {
        std::vector<bfloat16_t> src(static_cast<size_t>(rows) * K);
        for (int m = 0; m < rows; ++m)
            for (int k = 0; k < K; ++k)
                src[static_cast<size_t>(m) * K + k] = bfloat16_t(
                        0.01f * static_cast<float>((m * 11 + k * 3) % 13 - 6));
        return src;
    };
    auto prompt_src = make_src(prompt_M);
    auto decode_src = make_src(decode_M);
    std::vector<float> prompt_dst(static_cast<size_t>(prompt_M) * N);
    std::vector<bfloat16_t> decode_dst(static_cast<size_t>(decode_M) * N);

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::f32;
    params.num_threads = prompt_threads;
    matmul::matmul_batch_params_t batch;

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(2);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::NTileStrategyOverride rounds(2);
    auto &config = matmul_config_t::instance();
    const bool old_mixed = config.get_grp_auto_mixed_inplace();
    config.set_grp_auto_mixed_inplace(false);
    struct MixedRestore {
        matmul_config_t &config;
        bool value;
        ~MixedRestore() { config.set_grp_auto_mixed_inplace(value); }
    } mixed_restore {config, old_mixed};
    ::reset_grp_matmul_caches();

    ASSERT_EQ(matmul::matmul_direct('r', false, true, prompt_M, N, K, 1.0f,
                      prompt_src.data(), K, weight.data(), K,
                      /*bias=*/nullptr, 0.0f, prompt_dst.data(), N, true, batch,
                      params),
            status_t::success);
    params.dtypes.dst = data_type_t::bf16;
    params.num_threads = decode_threads;
    {
        mt::GemmModeCaptureGuard mode_capture;
        ASSERT_EQ(matmul::matmul_direct('r', false, true, decode_M, N, K, 1.0f,
                          decode_src.data(), K, weight.data(), K,
                          /*bias=*/nullptr, 0.0f, decode_dst.data(), N, true,
                          batch, params),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_gemm_mode.load(
                        std::memory_order_relaxed);
        ASSERT_NE(mode, nullptr);
        EXPECT_NE(std::string(mode).find("custom"), std::string::npos) << mode;
    }
    for (int m = 0; m < decode_M; ++m) {
        for (int n = 0; n < N; ++n) {
            float reference = 0.0f;
            for (int k = 0; k < K; ++k) {
                reference += static_cast<float>(
                                     decode_src[static_cast<size_t>(m) * K + k])
                        * static_cast<float>(
                                pristine_weight[static_cast<size_t>(n) * K
                                        + k]);
            }
            const float expected = static_cast<float>(bfloat16_t(reference));
            const float actual = static_cast<float>(
                    decode_dst[static_cast<size_t>(m) * N + n]);
            EXPECT_NEAR(actual, expected, std::abs(expected) * 0.02f + 1.0e-4f);
        }
    }
    ::reset_grp_matmul_caches();
}

// A BF16 prompt MatMul below decode's N-tile count (an embeddinggemma QKV:
// K 768, N 1280 on 32 threads is 20 NR64 tiles, not 64) routes to the CK
// prompt tile executor, raw and AOCL-prepacked, and matches the reference.
// With the knob off it keeps the standard MatMul path.
static void run_narrow_prompt_case(bool packed_input, bool knob_on) {
    constexpr int M = 256, K = 768, N = 1280, threads = 32;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<bfloat16_t> weight(static_cast<size_t>(N) * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            src[static_cast<size_t>(m) * K + k] = bfloat16_t(
                    0.01f * static_cast<float>((m * 11 + k * 3) % 13 - 6));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * K + k]
                    = bfloat16_t(logical_weight(k, n));

    aligned_ptr_t packed;
    const void *route_weight = weight.data();
    if (packed_input) {
        reorder::reorder_params_t prepack;
        prepack.is_prepack = true;
        prepack.num_threads = threads;
        prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
        prepack.prepack.wei_dtype = data_type_t::bf16;
        prepack.prepack.src_dtype = data_type_t::bf16;
        prepack.prepack.K = K;
        prepack.prepack.N = N;
        prepack.prepack.ldb = K;
        prepack.prepack.transposed = true;
        const size_t bytes = reorder::weight_prepack_size(prepack);
        ASSERT_GT(bytes, 0u);
        packed.reset(zendnnl_aligned_alloc(64, bytes));
        ASSERT_NE(packed, nullptr);
        ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
                status_t::success);
        route_weight = packed.get();
    }

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::NTileStrategyOverride rounds(2);
    mt::SingleExpertPromptTileOverride knob(knob_on ? 1 : 0);
    matmul::clear_aocl_ck_weight_cache();
    ::reset_grp_matmul_caches();

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.mem_format_b = packed_input ? 'r' : 'n';
    params.num_threads = threads;
    matmul::matmul_batch_params_t batch;
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);

    namespace pt = zendnnl::lowoha::matmul::prompt_tile;
    const long served0 = pt::test_api::s_calls_served.load();
    mt::GemmModeCaptureGuard mode_capture;
    ASSERT_EQ(matmul::matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      src.data(), K, route_weight, K, /*bias=*/nullptr, 0.0f,
                      dst.data(), N, true, batch, params),
            status_t::success);
    const long served = pt::test_api::s_calls_served.load() - served0;
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_gemm_mode.load(
                    std::memory_order_relaxed);
    if (knob_on) {
        EXPECT_EQ(served, 1);
        ASSERT_NE(mode, nullptr);
        EXPECT_NE(std::string(mode).find("ptile"), std::string::npos) << mode;
    } else {
        EXPECT_EQ(served, 0);
        EXPECT_EQ(mode, nullptr) << "standard MatMul path expected: " << mode;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float reference = 0.0f;
            for (int k = 0; k < K; ++k)
                reference += static_cast<float>(
                                     src[static_cast<size_t>(m) * K + k])
                        * static_cast<float>(
                                weight[static_cast<size_t>(n) * K + k]);
            const float expected = static_cast<float>(bfloat16_t(reference));
            const float actual
                    = static_cast<float>(dst[static_cast<size_t>(m) * N + n]);
            ASSERT_NEAR(actual, expected, std::abs(expected) * 0.02f + 1.0e-3f)
                    << "m=" << m << " n=" << n;
        }
    }
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, NarrowBf16PromptRoutesToPromptTile) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    for (const bool packed : {false, true}) {
        SCOPED_TRACE(packed ? "aocl-prepacked" : "raw");
        run_narrow_prompt_case(packed, /*knob_on=*/true);
        run_narrow_prompt_case(packed, /*knob_on=*/false);
    }
}

TEST(TestMatmulUnreorderRoute, PackedPromptStaysBlockedAndSkipsUnreorder) {
    // With the BF16 single-expert prompt tile off, prompt AUTO selects AOCL.
    // The packed bytes must flow directly to AOCL; the AOCL-derived final CK
    // cache remains cold.
    mt::SingleExpertPromptTileOverride prompt_tile_off(0);
    run_route_case(true, /*M=*/64, /*expect_custom=*/false,
            /*expect_derived_cache_hit=*/false);
}

TEST(TestMatmulUnreorderRoute, PackedPromptUsesPromptTileCk) {
    // By default a BF16 prompt runs the CK prompt tile executor: the packed
    // bytes are unreordered once into the derived CK cache.
    mt::SingleExpertPromptTileOverride prompt_tile_on(1);
    run_route_case(true, /*M=*/64, /*expect_custom=*/true,
            /*expect_derived_cache_hit=*/true);
}

TEST(TestMatmulUnreorderRoute, PackedTransposedPaddedLdbUsesOriginalStride) {
    run_route_case(true, /*M=*/8, /*expect_custom=*/true,
            /*expect_derived_cache_hit=*/true, /*trans_b=*/true,
            /*ldb_padding=*/7);
}

TEST(TestMatmulUnreorderRoute, PackedNonTransposedPaddedLdbUsesOriginalStride) {
    run_route_case(true, /*M=*/8, /*expect_custom=*/true,
            /*expect_derived_cache_hit=*/true, /*trans_b=*/false,
            /*ldb_padding=*/11);
}

TEST(TestMatmulUnreorderRoute, DirectPackedAlphaDeclineFallsBackToAocl) {
    run_route_case(true, /*M=*/8, /*expect_custom=*/false,
            /*expect_derived_cache_hit=*/false, /*trans_b=*/true,
            /*ldb_padding=*/0, /*weight_cache_mode=*/1,
            /*direct_group=*/true, /*alpha_value=*/2.0f);
}

TEST(TestMatmulUnreorderRoute, DirectPackedCacheOffFallsBackToAocl) {
    run_route_case(true, /*M=*/8, /*expect_custom=*/false,
            /*expect_derived_cache_hit=*/false, /*trans_b=*/true,
            /*ldb_padding=*/0, /*weight_cache_mode=*/0,
            /*direct_group=*/true, /*alpha_value=*/1.0f,
            /*probe_derived_cache=*/false);
}

TEST(TestMatmulUnreorderRoute, RawDirectReluUsesStandardPostopPath) {
    run_route_case(false, /*M=*/8, /*expect_custom=*/false,
            /*expect_derived_cache_hit=*/false, /*trans_b=*/true,
            /*ldb_padding=*/0, /*weight_cache_mode=*/1,
            /*direct_group=*/true, /*alpha_value=*/1.0f,
            /*probe_derived_cache=*/false, /*relu_postop=*/true);
}

#if ZENDNNL_DEPENDS_AOCLDLP
TEST(TestMatmulUnreorderRoute, StaticU8S8AoclPackedUsesCachedDerivedCkPath) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI custom kernel is unavailable";
    }

    constexpr int M = 5, K = 64, N = 256, threads = 4;
    constexpr bool trans_b = true;
    constexpr int ldb = K;
    std::vector<uint8_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    std::vector<float> bias(N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<uint8_t>(101 + (i * 7) % 43);
    for (int n = 0; n < N; ++n) {
        bias[n] = static_cast<float>((n % 7) - 3) * 0.0625f;
        for (int k = 0; k < K; ++k) {
            weight[static_cast<size_t>(n) * K + k]
                    = static_cast<int8_t>((n + k * 3) % 9 - 4);
        }
    }

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = threads;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::s8;
    prepack.prepack.src_dtype = data_type_t::u8;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = ldb;
    prepack.prepack.transposed = trans_b;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
            status_t::success);

    float src_scale = 0.015625f;
    int32_t src_zp = 121;
    float wei_scale = 0.03125f;
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.bias = data_type_t::f32;
    params.mem_format_b = 'r';
    params.lowoha_algo = matmul_algo_t::none;
    params.weight_cache_type = 1;
    params.num_threads = threads;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.src_zp.buff = &src_zp;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {1, 1};
    params.quant_params.wei_scale.buff = &wei_scale;
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, 1};
    matmul::matmul_post_op relu;
    relu.po_type = zendnnl::ops::post_op_type_t::relu;
    params.postop_ = {relu};
    matmul::matmul_batch_params_t batch;

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    ::reset_grp_matmul_caches();

    std::vector<bfloat16_t> dst_ck(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_aocl(static_cast<size_t>(M) * N);
    {
        mt::GemmModeCaptureGuard mode_capture;
        ASSERT_EQ(matmul::matmul_direct('r', false, trans_b, M, N, K, 1.0f,
                          src.data(), K, packed.get(), ldb, bias.data(), 0.0f,
                          dst_ck.data(), N, true, batch, params),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_gemm_mode.load(
                        std::memory_order_relaxed);
        ASSERT_NE(mode, nullptr);
        EXPECT_NE(std::string(mode).find("custom"), std::string::npos) << mode;
    }

    std::shared_ptr<void> derived_ck;
    bool cache_hit = false;
    ASSERT_EQ(matmul::get_or_cache_aocl_ck_weight_int8(packed.get(), K, N, ldb,
                      trans_b, threads, ck::plan_pack_nr(K, N),
                      /*interleave_split_halves=*/false, derived_ck, &cache_hit,
                      /*weight_cache_mode=*/1),
            status_t::success);
    EXPECT_TRUE(cache_hit);

    ::reset_grp_matmul_caches();
    matmul_config_t::instance().set_custom_kernel_route(false);
    ASSERT_EQ(matmul::matmul_direct('r', false, trans_b, M, N, K, 1.0f,
                      src.data(), K, packed.get(), ldb, bias.data(), 0.0f,
                      dst_aocl.data(), N, true, batch, params),
            status_t::success);
    for (size_t i = 0; i < dst_ck.size(); ++i) {
        const float actual = static_cast<float>(dst_ck[i]);
        const float expected = static_cast<float>(dst_aocl[i]);
        EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected))
                << "i=" << i;
    }
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, ScalarCkPrepackedWeightFailsClosed) {
    constexpr int M = 4, K = 64, N = 256;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));
    std::vector<bfloat16_t> weight(
            static_cast<size_t>(N) * K, bfloat16_t(0.5f));
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.prepack.algo = matmul_algo_t::moe_custom_kernel;
    prepack.prepack.wei_dtype = data_type_t::bf16;
    prepack.prepack.src_dtype = data_type_t::bf16;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = K;
    prepack.prepack.transposed = true;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
            status_t::success);

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.mem_format_b = 'r';
    params.lowoha_algo = matmul_algo_t::moe_custom_kernel;
    matmul::matmul_batch_params_t batch;

    RouteConfigGuard route_config;
    EXPECT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                      K, packed.get(), K, /*bias=*/nullptr, 0.0f, dst.data(), N,
                      true, batch, params),
            status_t::failure)
            << "scalar dispatch must not reinterpret CK-VNNI bytes as AOCL";
}

TEST(TestMatmulUnreorderRoute, ScalarFallbackFailsOnMutatedWeightConflict) {
    constexpr int M = 4, K = 64, N = 128;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));
    std::vector<bfloat16_t> weight(
            static_cast<size_t>(K) * N, bfloat16_t(0.5f));
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);

    RouteConfigGuard route_config;
    auto &config = matmul_config_t::instance();
    config.set_custom_kernel_route(false);
    WeightCacheGuard cache_mode(1);
    ::reset_grp_matmul_caches();
    {
        matmul::scoped_grp_auto_mutation_tracking grouped_scope(true);
        matmul::aocl_mark_weight_buffer_mutated(weight.data());
    }

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.num_threads = 4;
    matmul::matmul_batch_params_t batch;
    EXPECT_EQ(matmul::matmul_direct('r', /*transA=*/false, /*transB=*/false, M,
                      N, K, /*alpha=*/1.0f, src.data(), K, weight.data(), N,
                      /*bias=*/nullptr, /*beta=*/0.0f, dst.data(), N,
                      /*is_weights_const=*/true, batch, params),
            status_t::failure)
            << "the scalar boundary must propagate AOCL's refusal instead of "
               "returning success over a mutated blocked weight";
    ::reset_grp_matmul_caches();
}
#endif

TEST(TestMatmulUnreorderRoute,
        MixedActiveAoclPackedAndRawWc2PreservesPolicyAndNumerics) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    GTEST_SKIP() << "AOCL-DLP is unavailable";
#else
    constexpr int E = 2;
    constexpr int M = 8;
    constexpr int K = 64;
    constexpr int N = 2048;
    constexpr int threads = E;

    std::vector<std::vector<bfloat16_t>> src(E), raw_weight(E), dst(E);
    std::vector<const void *> src_ptr(E), weight_ptr(E), bias(E, nullptr);
    std::vector<void *> dst_ptr(E);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        raw_weight[e].resize(static_cast<size_t>(N) * K);
        dst[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                src[e][static_cast<size_t>(m) * K + k] = bfloat16_t(0.01f
                        * static_cast<float>((m * 11 + k * 3 + e) % 13 - 6));
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                raw_weight[e][static_cast<size_t>(n) * K + k]
                        = bfloat16_t(logical_weight(k, n + e));
        src_ptr[e] = src[e].data();
        weight_ptr[e] = raw_weight[e].data();
        dst_ptr[e] = dst[e].data();
    }
    const auto raw_expert_before = raw_weight[1];

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = threads;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::bf16;
    prepack.prepack.src_dtype = data_type_t::bf16;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = K;
    prepack.prepack.transposed = true;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(
                      raw_weight[0].data(), packed.get(), prepack),
            status_t::success);
    weight_ptr[0] = packed.get();

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(2);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::NTileStrategyOverride rounds(2);
    auto &config = matmul_config_t::instance();
    const bool old_mixed = config.get_grp_auto_mixed_inplace();
    config.set_grp_auto_mixed_inplace(false);
    struct MixedRestore {
        matmul_config_t &config;
        bool value;
        ~MixedRestore() { config.set_grp_auto_mixed_inplace(value); }
    } mixed_restore {config, old_mixed};
    ::reset_grp_matmul_caches();

    std::vector<char> layout(E, 'r');
    std::vector<bool> trans_a(E, false), trans_b(E, true);
    std::vector<int> ms(E, M), ns(E, N), ks(E, K), lda(E, K), ldb(E, K),
            ldc(E, N);
    std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    std::vector<bool> weights_const(E, true);
    std::vector<matmul::matmul_params> params(E);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
        p.num_threads = threads;
    }
    params[0].mem_format_b = 'r';
    params[0].lowoha_algo = matmul_algo_t::none;

    mt::GemmModeCaptureGuard mode_capture;
    ASSERT_EQ(matmul::group_matmul_direct(layout, trans_a, trans_b, ms, ns, ks,
                      alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst_ptr,
                      ldc, weights_const, params, /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_gemm_mode.load(
                    std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_NE(std::string(mode).find("custom"), std::string::npos);
    EXPECT_EQ(config.get_weight_cache(), 2);
    EXPECT_TRUE(config.get_grp_auto_mixed_inplace());
    EXPECT_EQ(std::memcmp(raw_weight[1].data(), raw_expert_before.data(),
                      raw_expert_before.size() * sizeof(bfloat16_t)),
            0)
            << "the raw expert in a mixed pool must stay out-of-place";

    for (int e = 0; e < E; ++e) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float reference = 0.0f;
                for (int k = 0; k < K; ++k) {
                    reference += static_cast<float>(
                                         src[e][static_cast<size_t>(m) * K + k])
                            * static_cast<float>(
                                    bfloat16_t(logical_weight(k, n + e)));
                }
                const float expected
                        = static_cast<float>(bfloat16_t(reference));
                const float actual = static_cast<float>(
                        dst[e][static_cast<size_t>(m) * N + n]);
                EXPECT_NEAR(
                        actual, expected, std::abs(expected) * 0.02f + 1.0e-4f);
            }
        }
    }
    ::reset_grp_matmul_caches();
#endif
}

TEST(TestMatmulUnreorderRoute,
        RejectsForeignPackedProvenanceBeforeFallbackDispatch) {
    constexpr int E = 1, M = 2, K = 64, N = 64;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));
    std::vector<bfloat16_t> foreign_packed(
            static_cast<size_t>(K) * N, bfloat16_t(0.5f));
    const bfloat16_t canary(7.0f);
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N, canary);
    std::vector<const void *> src_ptr = {src.data()};
    std::vector<const void *> weight_ptr = {foreign_packed.data()};
    std::vector<const void *> bias = {nullptr};
    std::vector<void *> dst_ptr = {dst.data()};
    std::vector<char> layout(E, 'r');
    std::vector<bool> trans_a(E, false), trans_b(E, false);
    std::vector<int> ms(E, M), ns(E, N), ks(E, K), lda(E, K), ldb(E, N),
            ldc(E, N);
    std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    std::vector<bool> weights_const(E, true);
    std::vector<matmul::matmul_params> params(E);
    params[0].dtypes.src = data_type_t::bf16;
    params[0].dtypes.wei = data_type_t::bf16;
    params[0].dtypes.dst = data_type_t::bf16;
    params[0].mem_format_b = 'r';
    params[0].lowoha_algo = matmul_algo_t::onednn_blocked;

    mt::AlgoEnvGuard algo1(1);
    EXPECT_EQ(matmul::group_matmul_direct(layout, trans_a, trans_b, ms, ns, ks,
                      alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst_ptr,
                      ldc, weights_const, params, /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::failure);
    for (const bfloat16_t value : dst) {
        EXPECT_FLOAT_EQ(static_cast<float>(value), static_cast<float>(canary));
    }
}

TEST(TestMatmulUnreorderRoute, RejectsForeignPackedProvenanceAtScalarBoundary) {
    constexpr int M = 2, K = 64, N = 64;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));
    std::vector<bfloat16_t> foreign_packed(
            static_cast<size_t>(K) * N, bfloat16_t(0.5f));
    const bfloat16_t canary(7.0f);
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N, canary);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.mem_format_b = 'r';
    params.lowoha_algo = matmul_algo_t::onednn_blocked;
    matmul::matmul_batch_params_t batch;

    EXPECT_EQ(matmul::matmul_direct('r', /*transA=*/false, /*transB=*/false, M,
                      N, K, /*alpha=*/1.0f, src.data(), /*lda=*/K,
                      foreign_packed.data(), /*ldb=*/N, /*bias=*/nullptr,
                      /*beta=*/0.0f, dst.data(), /*ldc=*/N,
                      /*is_weights_const=*/true, batch, params),
            status_t::failure);
    for (const bfloat16_t value : dst) {
        EXPECT_FLOAT_EQ(static_cast<float>(value), static_cast<float>(canary));
    }
}

#if !ZENDNNL_DEPENDS_AOCLDLP
TEST(TestMatmulUnreorderRoute, RejectsAoclBlockedFallbackWithoutAocl) {
    constexpr int E = 1, M = 2, K = 64, N = 64;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));
    std::vector<bfloat16_t> packed_bytes(
            static_cast<size_t>(K) * N, bfloat16_t(0.5f));
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
    std::vector<const void *> src_ptr = {src.data()};
    std::vector<const void *> weight_ptr = {packed_bytes.data()};
    std::vector<const void *> bias = {nullptr};
    std::vector<void *> dst_ptr = {dst.data()};
    std::vector<char> layout(E, 'r');
    std::vector<bool> trans_a(E, false), trans_b(E, false);
    std::vector<int> ms(E, M), ns(E, N), ks(E, K), lda(E, K), ldb(E, N),
            ldc(E, N);
    std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    std::vector<bool> weights_const(E, true);
    std::vector<matmul::matmul_params> params(E);
    params[0].dtypes.src = data_type_t::bf16;
    params[0].dtypes.wei = data_type_t::bf16;
    params[0].dtypes.dst = data_type_t::bf16;
    params[0].mem_format_b = 'r';
    // Legacy scalar-prepack representation: mem_format_b carries AOCL
    // provenance while lowoha_algo remains unset.
    params[0].lowoha_algo = matmul_algo_t::none;

    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_off(false);
    EXPECT_EQ(matmul::group_matmul_direct(layout, trans_a, trans_b, ms, ns, ks,
                      alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst_ptr,
                      ldc, weights_const, params, /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::failure)
            << "an AOCL-disabled build must reject AOCL-blocked bytes rather "
               "than pass them to the raw reference kernel";
}
#endif

// One raw narrow-N weight (below decode's N-tile count) goes prompt -> decode
// -> prompt through `matmul_direct`.  With the single-expert prompt tile knob
// on, both phases route to the custom kernel, so a WEIGHT_CACHE=2 in-place
// pack is the custom-kernel layout and every later call reads it; the buffer
// never holds an AOCL layout.  With the knob off, the weight stays raw.
void run_narrow_phase_switch(int weight_cache_mode, bool with_bias, int knob,
        int algo_pin = 0, uint32_t lru_capacity = 0) {
    constexpr int N = 1280, K = 768, threads = 32;
    std::vector<bfloat16_t> weight(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * K + k]
                    = bfloat16_t(logical_weight(k, n));
    const std::vector<bfloat16_t> weight_copy = weight;
    std::vector<float> bias(N);
    for (int n = 0; n < N; ++n)
        bias[n] = 0.01f * static_cast<float>(n % 7 - 3);

    RouteConfigGuard route_config;
    if (lru_capacity != 0)
        matmul_config_t::instance().set_lru_cache_capacity(lru_capacity);
    WeightCacheGuard cache_mode(weight_cache_mode);
    mt::AlgoEnvGuard algo(algo_pin);
    mt::CustomKernelOverride custom_on(true);
    mt::SingleExpertPromptTileOverride knob_guard(knob);
    matmul::clear_aocl_ck_weight_cache();
    ::reset_grp_matmul_caches();
    namespace pt = zendnnl::lowoha::matmul::prompt_tile;
    const long served0 = pt::test_api::s_calls_served.load();

    for (const int M : {256, 4, 32, 33, 100, 1, 512}) {
        SCOPED_TRACE("M=" + std::to_string(M));
        std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                src[static_cast<size_t>(m) * K + k] = bfloat16_t(
                        0.01f * static_cast<float>((m * 11 + k * 3) % 13 - 6));
        matmul::matmul_params params;
        params.dtypes.src = data_type_t::bf16;
        params.dtypes.wei = data_type_t::bf16;
        params.dtypes.dst = data_type_t::bf16;
        params.dtypes.bias = with_bias ? data_type_t::f32 : data_type_t::none;
        params.num_threads = threads;
        matmul::matmul_batch_params_t batch;
        std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
        ASSERT_EQ(matmul::matmul_direct('r', false, /*transB=*/true, M, N, K,
                          1.0f, src.data(), K, weight.data(), K,
                          with_bias ? bias.data() : nullptr, 0.0f, dst.data(),
                          N, true, batch, params),
                status_t::success);
        double max_err = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float ref = with_bias ? bias[n] : 0.0f;
                for (int k = 0; k < K; ++k)
                    ref += static_cast<float>(
                                   src[static_cast<size_t>(m) * K + k])
                            * static_cast<float>(
                                    weight_copy[static_cast<size_t>(n) * K
                                            + k]);
                const float got = static_cast<float>(
                        dst[static_cast<size_t>(m) * N + n]);
                max_err = std::max(max_err,
                        static_cast<double>(std::abs(got - ref)
                                / (std::abs(ref) * 0.02f + 1.0e-3f)));
            }
        EXPECT_LE(max_err, 1.0) << "normalised error";
    }
    const bool mutated = std::memcmp(weight.data(), weight_copy.data(),
                                 weight.size() * sizeof(bfloat16_t))
            != 0;
    if (knob != 0) {
        EXPECT_GT(pt::test_api::s_calls_served.load() - served0, 0)
                << "prompts must run the prompt tile";
        EXPECT_TRUE(!mutated || ck::weight_packed_in_place(weight.data()))
                << "an in-place image must be the custom-kernel pack";
    } else {
        EXPECT_FALSE(mutated) << "knob off keeps the raw weight";
    }
    ::reset_grp_matmul_caches();
    matmul::clear_aocl_ck_weight_cache();
}

TEST(TestMatmulUnreorderRoute, NarrowWeightPromptDecodeShareCkPack) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    for (const int knob : {1, 0})
        for (const int wc : {1, 2})
            for (const bool bias : {false, true}) {
                SCOPED_TRACE("knob=" + std::to_string(knob) + " wc="
                        + std::to_string(wc) + " bias=" + std::to_string(bias));
                run_narrow_phase_switch(wc, bias, knob);
            }
}

// The WEIGHT_CACHE=2 settings that turn AUTO mixed in-place mode off (an
// ALGO 3 pin, a finite LRU capacity) let the prompt's pack go in place; the
// decode must then read it through the custom kernel.
TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackWithoutMixedMode) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    for (const bool bias : {false, true}) {
        SCOPED_TRACE("algo3 pin, bias=" + std::to_string(bias));
        run_narrow_phase_switch(2, bias, 1, /*algo_pin=*/3);
    }
    for (const bool bias : {false, true}) {
        SCOPED_TRACE("LRU capacity 64, bias=" + std::to_string(bias));
        run_narrow_phase_switch(2, bias, 1, 0, /*lru_capacity=*/64);
    }
}

// CROSS_WARM and PREPACK are read once per process, so each runs in a child.
void expect_narrow_phase_switch_in_child(const char *env, const char *value) {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    setenv(env, value, /*overwrite=*/1);
    EXPECT_EXIT(
            {
                run_narrow_phase_switch(2, /*with_bias=*/false, 1);
                run_narrow_phase_switch(2, /*with_bias=*/true, 1);
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    unsetenv(env);
}

TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackCrossWarmOff) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    expect_narrow_phase_switch_in_child("ZENDNNL_GRP_MATMUL_CROSS_WARM", "0");
}

TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackPrepackOff) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    expect_narrow_phase_switch_in_child("ZENDNNL_GRP_MATMUL_PREPACK", "0");
}

// `decode_d` (N_TILE_STRATEGY=1) never reaches the prompt tile executor, so a
// narrow BF16 prompt keeps the standard MatMul path instead of DecodeD.
TEST(TestMatmulUnreorderRoute, NarrowBf16PromptDecodeDKeepsStandardPath) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::SingleExpertPromptTileOverride knob(1);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    const auto routable = [&](int M) {
        return matmul::custom_kernel_routable('r', M, 1280, 768, 32, 1, false,
                true, 1.0f, 0.0f, nullptr, 768, true, params);
    };
    EXPECT_TRUE(routable(2048));
    EXPECT_TRUE(routable(4)) << "a raw weight's decode follows its prompt";
    mt::NTileStrategyOverride decode_d(1);
    EXPECT_FALSE(routable(2048));
    EXPECT_FALSE(routable(4));
}

} // namespace
