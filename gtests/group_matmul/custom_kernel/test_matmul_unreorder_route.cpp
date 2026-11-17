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
#include "common/float16.hpp"
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_route.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "moe_test_utils.hpp"

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
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
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (packed_input) GTEST_SKIP() << "AOCL-DLP is unavailable";
#endif
    // An earlier test's pack can still be registered at an address this
    // case's fresh buffers reuse.
    ::reset_grp_matmul_caches();

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

    mt::ExecutionPathCaptureGuard mode_capture;
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
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
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
    // M == kDecodeMaxM is decode, so the prompt uses kDecodeMaxM + 1.
    for (const bool prompt_tile : {false, true}) {
        SCOPED_TRACE(prompt_tile ? "prompt tile on" : "prompt tile off");
        mt::SingleExpertCkPolicy knob(
                prompt_tile ? 1 : 0, prompt_tile ? 1 : 0, 0);
        run_route_case(/*packed_input=*/true, /*M=*/matmul::kDecodeMaxM + 1,
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

TEST(TestMatmulUnreorderRoute, F16PackedWeightRoutesAndReusesDerivedCkCache) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    GTEST_SKIP() << "AOCL-DLP is unavailable";
#else
    if (!ck::avx512f16_available()) {
        GTEST_SKIP() << "AVX-512-FP16 custom kernel is unavailable";
    }

    constexpr int M = 8;
    constexpr int K = 64;
    constexpr int N = 2048;
    constexpr int threads = 8;
    constexpr int ldb = K;

    std::vector<float16_t> src(static_cast<size_t>(M) * K);
    std::vector<float16_t> weight(static_cast<size_t>(N) * ldb);
    std::vector<float> bias(N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            src[static_cast<size_t>(m) * K + k] = float16_t(
                    0.01f * static_cast<float>((m * 11 + k * 3) % 13 - 6));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * ldb + k]
                    = float16_t(logical_weight(k, n));
    for (int n = 0; n < N; ++n)
        bias[n] = 0.001f * static_cast<float>(n % 7 - 3);

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = threads;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::f16;
    prepack.prepack.src_dtype = data_type_t::f16;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = ldb;
    prepack.prepack.transposed = true;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
            status_t::success);

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard auto_algo(0);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelF16Override custom_f16_on(true);
    mt::NTileStrategyOverride rounds(2);
    mt::SingleExpertCkMasks f16_policy(matmul::kSingleExpertF16,
            matmul::kSingleExpertF16, matmul::kSingleExpertF16);
    matmul::clear_aocl_ck_weight_cache();

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::f16;
    params.dtypes.wei = data_type_t::f16;
    params.dtypes.dst = data_type_t::f16;
    params.dtypes.bias = data_type_t::f32;
    params.mem_format_b = 'r';
    params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    params.num_threads = threads;
    matmul::matmul_batch_params_t batch;

    std::shared_ptr<void> first_cached_weight;
    const auto tolerance = mt::tol_act(data_type_t::f16);
    for (int call = 0; call < 2; ++call) {
        std::vector<float16_t> dst(static_cast<size_t>(M) * N);
        {
            mt::ExecutionPathCaptureGuard mode_capture;
            ASSERT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f,
                              src.data(), K, packed.get(), ldb, bias.data(),
                              0.0f, dst.data(), N, true, batch, params),
                    status_t::success);
            const char *mode
                    = matmul::test_api::
                              s_last_group_matmul_direct_execution_path.load(
                                      std::memory_order_relaxed);
            ASSERT_NE(mode, nullptr);
            EXPECT_NE(std::string(mode).find("custom"), std::string::npos);
        }

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float reference = bias[n];
                for (int k = 0; k < K; ++k) {
                    reference += static_cast<float>(
                                         src[static_cast<size_t>(m) * K + k])
                            * static_cast<float>(
                                    weight[static_cast<size_t>(n) * ldb + k]);
                }
                const float actual = static_cast<float>(
                        dst[static_cast<size_t>(m) * N + n]);
                EXPECT_NEAR(actual, reference,
                        std::abs(reference) * tolerance.rel + tolerance.abs);
            }
        }

        std::shared_ptr<void> cached_weight;
        bool hit = false;
        ASSERT_EQ(
                matmul::get_or_cache_aocl_ck_weight_f16(packed.get(), K, N, ldb,
                        /*transB=*/true, threads, ck::plan_pack_nr(K, N),
                        /*interleave_split_halves=*/false, cached_weight, &hit),
                status_t::success);
        ASSERT_TRUE(hit) << "matmul_direct must populate and then reuse the "
                            "derived cache";
        ASSERT_NE(cached_weight, nullptr);
        if (call == 0) {
            first_cached_weight = cached_weight;
        } else {
            EXPECT_EQ(cached_weight.get(), first_cached_weight.get());
        }
    }

    first_cached_weight.reset();
    ::reset_grp_matmul_caches();
#endif
}

TEST(TestMatmulUnreorderRoute, RawWeightUsesDirectGroupPath) {
    run_route_case(false);
}

TEST(TestMatmulUnreorderRoute, RawWc2IneligiblePromptPreservesFutureDecode) {
    if (!ck::bf16_isa_supported()) {
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
        mt::ExecutionPathCaptureGuard mode_capture;
        ASSERT_EQ(matmul::matmul_direct('r', false, true, decode_M, N, K, 1.0f,
                          decode_src.data(), K, weight.data(), K,
                          /*bias=*/nullptr, 0.0f, decode_dst.data(), N, true,
                          batch, params),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_execution_path
                          .load(std::memory_order_relaxed);
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
// K 768, N 1280 on 32 threads is 40 NR32 tiles, not 64) routes to the CK
// prompt tile executor under FORCE_CK, raw and AOCL-prepacked, and matches
// the reference.  With FORCE_CK off it keeps the standard MatMul path.
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
    mt::SingleExpertCkPolicy knob(
            knob_on ? 1 : 0, knob_on ? 1 : 0, knob_on ? 1 : 0);
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
    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      src.data(), K, route_weight, K, /*bias=*/nullptr, 0.0f,
                      dst.data(), N, true, batch, params),
            status_t::success);
    const long served = pt::test_api::s_calls_served.load() - served0;
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
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
    if (!ck::bf16_isa_supported()) {
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
    mt::SingleExpertCkPolicy prompt_tile_off(0, 0, 0);
    run_route_case(true, /*M=*/matmul::kDecodeMaxM + 1, /*expect_custom=*/false,
            /*expect_derived_cache_hit=*/false);
}

TEST(TestMatmulUnreorderRoute, PackedPromptUsesPromptTileCk) {
    // By default a BF16 prompt runs the CK prompt tile executor: the packed
    // bytes are unreordered once into the derived CK cache.
    mt::SingleExpertCkPolicy prompt_tile_on(1, 1, 0);
    run_route_case(true, /*M=*/matmul::kDecodeMaxM + 1, /*expect_custom=*/true,
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

    constexpr int M = 4, K = 64, N = 256, threads = 4;
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
        mt::ExecutionPathCaptureGuard mode_capture;
        ASSERT_EQ(matmul::matmul_direct('r', false, trans_b, M, N, K, 1.0f,
                          src.data(), K, packed.get(), ldb, bias.data(), 0.0f,
                          dst_ck.data(), N, true, batch, params),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_execution_path
                          .load(std::memory_order_relaxed);
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

void run_symmetric_quant_aocl_packed_route(
        bool s4, bool pending_dynamic_scale = false) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI custom kernel is unavailable";
    }

    constexpr int K = 64, N = 256, threads = 4;
    const int M = pending_dynamic_scale ? 1 : 5;
    const int group_size = s4 ? 16 : K;
    const int G = K / group_size;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    const size_t weight_elems = static_cast<size_t>(N) * K;
    std::vector<int8_t> weight(s4 ? (weight_elems + 1) / 2 : weight_elems, 0);
    std::vector<float> src_scale(M);
    std::vector<float> wei_scale(static_cast<size_t>(G) * N);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = bfloat16_t(
                0.03125f * static_cast<float>(static_cast<int>(i % 17) - 8));
    for (int m = 0; m < M; ++m)
        src_scale[m] = 0.0125f + 0.0005f * static_cast<float>(m);
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            wei_scale[static_cast<size_t>(g) * N + n]
                    = 0.02f + 0.00025f * static_cast<float>((g + n) % 5);
    for (size_t i = 0; i < weight_elems; ++i) {
        const int value = static_cast<int>((i * 7 + 1) % (s4 ? 15 : 23))
                - (s4 ? 7 : 11);
        if (s4) {
            uint8_t &byte = reinterpret_cast<uint8_t *>(weight.data())[i >> 1];
            const uint8_t nibble = static_cast<uint8_t>(value) & 0x0F;
            byte = (i & 1U) ? static_cast<uint8_t>((byte & 0x0F) | nibble << 4)
                            : static_cast<uint8_t>((byte & 0xF0) | nibble);
        } else {
            weight[i] = static_cast<int8_t>(value);
        }
    }

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = threads;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = s4 ? data_type_t::s4 : data_type_t::s8;
    prepack.prepack.src_dtype = data_type_t::s8;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = K;
    prepack.prepack.transposed = true;
    prepack.prepack.sym_group_size = group_size;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    if (packed_bytes == 0) {
        GTEST_SKIP() << "AOCL symmetric quantized reorder is unavailable";
    }
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
            status_t::success);

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = s4 ? data_type_t::s4 : data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.mem_format_b = 'r';
    params.weight_cache_type = 1;
    params.num_threads = threads;
    params.quant_params.src_scale.buff
            = pending_dynamic_scale ? nullptr : src_scale.data();
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {G, N};
    matmul::matmul_batch_params_t batch;

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard auto_algo(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    const unsigned family
            = s4 ? matmul::kSingleExpertW4a8 : matmul::kSingleExpertInt8;
    mt::SingleExpertCkMasks single_expert(family, family, family);
    ::reset_grp_matmul_caches();

    std::vector<bfloat16_t> dst_ck(static_cast<size_t>(M) * N);
    std::vector<bfloat16_t> dst_aocl(static_cast<size_t>(M) * N);
    const auto variant = ck::resolve_variant(params.dtypes.src,
            params.dtypes.wei, params.dtypes.dst, params.dynamic_quant,
            params.dtypes.compute);
    ASSERT_NE(variant, ck::KernelVariant::kUnsupported);
    const auto single_ck
            = matmul::grp_matmul_single_expert_ck(params, K, N, threads);
    ASSERT_TRUE(single_ck.decode);
    ASSERT_TRUE(matmul::matmul_custom_kernel_quant_supported(
            M, N, K, variant, params));
    ASSERT_TRUE(matmul::group_matmul_custom_kernel_candidate(
            std::vector<matmul::matmul_params> {params}, std::vector<int> {M},
            std::vector<int> {N}, matmul::grp_matmul_gated_act_t::none));
    const int selected = matmul::group_matmul_algo_select(
            std::vector<char> {'r'}, std::vector<int> {M}, std::vector<int> {N},
            std::vector<int> {K}, std::vector<matmul::matmul_params> {params},
            threads);
    ASSERT_EQ(selected, 3);
    ASSERT_TRUE(matmul::single_matmul_group_routable('r', M, N, K, threads,
            /*batch_count=*/1, /*transA=*/false, /*transB=*/true,
            /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/nullptr, /*ldb=*/K,
            /*is_weights_const=*/true, params));
    {
        mt::ExecutionPathCaptureGuard mode_capture;
        ASSERT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f,
                          src.data(), K, packed.get(), K, /*bias=*/nullptr,
                          0.0f, dst_ck.data(), N, true, batch, params),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_execution_path
                          .load(std::memory_order_relaxed);
        ASSERT_NE(mode, nullptr);
        EXPECT_NE(std::string(mode).find("custom"), std::string::npos) << mode;
    }

    std::shared_ptr<void> derived_ck;
    bool hit = false;
    const status_t cache_status = s4
            ? matmul::get_or_cache_aocl_ck_weight_s4(packed.get(), K, N, K,
                      /*transB=*/true, threads, ck::plan_pack_nr(K, N),
                      /*interleave_split_halves=*/false, group_size, derived_ck,
                      &hit, /*weight_cache_mode=*/1)
            : matmul::get_or_cache_aocl_ck_weight_int8_sym(packed.get(), K, N,
                      K, /*transB=*/true, threads, ck::plan_pack_nr(K, N),
                      /*interleave_split_halves=*/false, group_size, derived_ck,
                      &hit, /*weight_cache_mode=*/1);
    ASSERT_EQ(cache_status, status_t::success);
    EXPECT_TRUE(hit);

    ::reset_grp_matmul_caches();
    matmul_config_t::instance().set_custom_kernel_route(false);
    ASSERT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                      K, packed.get(), K, /*bias=*/nullptr, 0.0f,
                      dst_aocl.data(), N, true, batch, params),
            status_t::success);
    for (size_t i = 0; i < dst_ck.size(); ++i) {
        const float actual = static_cast<float>(dst_ck[i]);
        const float expected = static_cast<float>(dst_aocl[i]);
        EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected))
                << "i=" << i;
    }
    derived_ck.reset();
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, SymmetricInt8AoclPackedRoutesToCk) {
    run_symmetric_quant_aocl_packed_route(/*s4=*/false);
}

TEST(TestMatmulUnreorderRoute, NativeS4AoclPackedRoutesToCk) {
    run_symmetric_quant_aocl_packed_route(/*s4=*/true);
}

TEST(TestMatmulUnreorderRoute, SymmetricInt8PendingDynamicScaleRoutesToCk) {
    mt::GroupDqOverride group_dq_off(0);
    run_symmetric_quant_aocl_packed_route(
            /*s4=*/false, /*pending_dynamic_scale=*/true);
}

TEST(TestMatmulUnreorderRoute, DynamicAsymmetricAoclPackedKeepsAoclFallback) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI is unavailable";
    }
    constexpr int M = 4, K = 64, N = 256, threads = 4;
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = bfloat16_t(
                0.025f * static_cast<float>(static_cast<int>(i % 17) - 8));
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 23) - 11);

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::s8;
    prepack.prepack.src_dtype = data_type_t::u8;
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

    std::vector<float> src_scale(M, 0.0f), wei_scale(N, 0.0125f);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.compute = data_type_t::u8;
    params.dynamic_quant = true;
    params.mem_format_b = 'r';
    params.weight_cache_type = 1;
    params.num_threads = threads;
    params.quant_params.src_scale.buff = src_scale.data();
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.src_zp.buff = nullptr;
    params.quant_params.src_zp.dt = data_type_t::s32;
    params.quant_params.src_zp.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::GroupDqOverride group_dq_off(0);
    mt::NTileStrategyOverride rounds(2);
    matmul::matmul_batch_params_t batch;
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                      K, packed.get(), K, /*bias=*/nullptr, 0.0f, dst.data(), N,
                      true, batch, params),
            status_t::success);
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
                    std::memory_order_relaxed);
    EXPECT_EQ(mode, nullptr)
            << "asymmetric AOCL-prepacked INT8 should stay on scalar matmul";
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, PlainS8AoclPackDoesNotUseSymmetricInverse) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI is unavailable";
    }
    constexpr int M = 1, K = 64, N = 256, threads = 4;
    std::vector<int8_t> src(static_cast<size_t>(M) * K);
    std::vector<int8_t> weight(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<int8_t>(static_cast<int>(i % 17) - 8);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int8_t>(static_cast<int>(i % 23) - 11);

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::s8;
    prepack.prepack.src_dtype = data_type_t::s8;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = K;
    prepack.prepack.transposed = true;
    prepack.prepack.sym_group_size = 0;
    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(weight.data(), packed.get(), prepack),
            status_t::success);

    float src_scale = 0.02f;
    std::vector<float> wei_scale(N, 0.01f);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::s8;
    params.dtypes.wei = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.dtypes.compute = data_type_t::s8;
    params.mem_format_b = 'r';
    params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    params.weight_cache_type = 1;
    params.num_threads = threads;
    params.quant_params.src_scale.buff = &src_scale;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {1, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {N};

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    matmul::matmul_batch_params_t batch;
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                      K, packed.get(), K, /*bias=*/nullptr, 0.0f, dst.data(), N,
                      true, batch, params),
            status_t::success);
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
                    std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(std::string(mode).find("custom"), std::string::npos) << mode;
    ::reset_grp_matmul_caches();
}

void run_multi_expert_prepacked_int8(
        bool heterogeneous_k, bool ck_prepacked = false) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI custom kernel is unavailable";
    }

    constexpr int E = 3, rows = 4, N = 256, threads = 4;
    const std::vector<int> M {0, rows, rows};
    const std::vector<int> K {32, 32, heterogeneous_k ? 128 : 32};
    std::vector<std::vector<bfloat16_t>> src(E);
    std::vector<std::vector<int8_t>> raw_weight(E);
    std::vector<std::vector<float>> src_scale(E), wei_scale(E);
    std::vector<aligned_ptr_t> packed(E);
    std::vector<const void *> src_ptr(E, nullptr), weight_ptr(E, nullptr),
            ref_weight_ptr(E, nullptr);
    std::vector<void *> dst_ptr(E, nullptr), ref_ptr(E, nullptr);
    std::vector<std::vector<bfloat16_t>> dst(E), ref(E);
    std::vector<matmul::matmul_params> params(E);

    for (int e = 0; e < E; ++e) {
        params[e].dtypes.src = data_type_t::bf16;
        params[e].dtypes.wei = data_type_t::s8;
        params[e].dtypes.dst = data_type_t::bf16;
        params[e].dtypes.compute = data_type_t::s8;
        params[e].dynamic_quant = true;
        params[e].mem_format_b = 'r';
        params[e].lowoha_algo = ck_prepacked ? matmul_algo_t::moe_custom_kernel
                                             : matmul_algo_t::aocl_dlp_blocked;
        params[e].weight_cache_type = 1;
        params[e].num_threads = threads;
        params[e].quant_params.src_scale.dt = data_type_t::f32;
        params[e].quant_params.src_scale.dims
                = {static_cast<int64_t>(std::max(1, M[e])), 1};
        params[e].quant_params.wei_scale.dt = data_type_t::f32;
        params[e].quant_params.wei_scale.dims = {1, N};
        if (M[e] == 0) continue;

        src[e].resize(static_cast<size_t>(M[e]) * K[e]);
        raw_weight[e].resize(static_cast<size_t>(N) * K[e]);
        src_scale[e].resize(M[e], 0.0f);
        wei_scale[e].resize(N);
        dst[e].resize(static_cast<size_t>(M[e]) * N);
        ref[e].resize(static_cast<size_t>(M[e]) * N);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = bfloat16_t(0.025f
                    * static_cast<float>(
                            static_cast<int>((i + e * 3) % 19) - 9));
        for (size_t i = 0; i < raw_weight[e].size(); ++i)
            raw_weight[e][i] = static_cast<int8_t>(
                    static_cast<int>((i * 5 + e) % 23) - 11);
        for (int n = 0; n < N; ++n)
            wei_scale[e][n]
                    = 0.0125f + 0.00025f * static_cast<float>((n + e) % 7);

        reorder::reorder_params_t prepack;
        prepack.is_prepack = true;
        prepack.prepack.algo = ck_prepacked ? matmul_algo_t::moe_custom_kernel
                                            : matmul_algo_t::aocl_dlp_blocked;
        prepack.prepack.wei_dtype = data_type_t::s8;
        prepack.prepack.src_dtype = data_type_t::s8;
        prepack.prepack.K = K[e];
        prepack.prepack.N = N;
        prepack.prepack.ldb = K[e];
        prepack.prepack.transposed = true;
        prepack.prepack.sym_group_size = ck_prepacked ? 0 : K[e];
        const size_t packed_bytes = reorder::weight_prepack_size(prepack);
        ASSERT_GT(packed_bytes, 0u);
        packed[e].reset(zendnnl_aligned_alloc(64, packed_bytes));
        ASSERT_NE(packed[e], nullptr);
        ASSERT_EQ(reorder::reorder_direct(
                          raw_weight[e].data(), packed[e].get(), prepack),
                status_t::success);

        params[e].quant_params.src_scale.buff = src_scale[e].data();
        params[e].quant_params.wei_scale.buff = wei_scale[e].data();
        src_ptr[e] = src[e].data();
        weight_ptr[e] = packed[e].get();
        ref_weight_ptr[e] = raw_weight[e].data();
        dst_ptr[e] = dst[e].data();
        ref_ptr[e] = ref[e].data();
    }

    const std::vector<char> layout(E, 'r');
    const std::vector<bool> transA(E, false), transB(E, true);
    const std::vector<int> N_vec(E, N), lda = K, ldb = K, ldc(E, N);
    const std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    const std::vector<const void *> bias(E, nullptr);
    const std::vector<bool> weights_const(E, true);

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    ::reset_grp_matmul_caches();

    {
        mt::ExecutionPathCaptureGuard mode_capture;
        ASSERT_EQ(matmul::group_matmul_direct(layout, transA, transB, M, N_vec,
                          K, alpha, src_ptr, lda, weight_ptr, ldb, bias, beta,
                          dst_ptr, ldc, weights_const, params,
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success);
        const char *mode
                = matmul::test_api::s_last_group_matmul_direct_execution_path
                          .load(std::memory_order_relaxed);
        ASSERT_NE(mode, nullptr);
        if (heterogeneous_k && !ck_prepacked) {
            EXPECT_EQ(std::string(mode).find("custom"), std::string::npos)
                    << mode;
        } else {
            EXPECT_NE(std::string(mode).find("custom"), std::string::npos)
                    << mode;
        }
    }

    ::reset_grp_matmul_caches();
    matmul_config_t::instance().set_custom_kernel_route(false);
    std::vector<matmul::matmul_params> ref_params = params;
    for (auto &p : ref_params) {
        p.mem_format_b = 'n';
        p.lowoha_algo = matmul_algo_t::none;
    }
    {
        mt::AlgoEnvGuard algo1(1);
        ASSERT_EQ(matmul::group_matmul_direct(layout, transA, transB, M, N_vec,
                          K, alpha, src_ptr, lda, ref_weight_ptr, ldb, bias,
                          beta, ref_ptr, ldc, weights_const, ref_params,
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success);
    }
    for (int e = 1; e < E; ++e) {
        for (size_t i = 0; i < dst[e].size(); ++i) {
            const float expected = static_cast<float>(ref[e][i]);
            const float actual = static_cast<float>(dst[e][i]);
            EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected))
                    << "expert=" << e << " i=" << i;
        }
    }
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, MultiExpertSymmetricInt8AoclPackedRoutesToCk) {
    run_multi_expert_prepacked_int8(/*heterogeneous_k=*/false);
}

TEST(TestMatmulUnreorderRoute,
        HeterogeneousAoclPackedGroupSizesFallBackAtFullN) {
    run_multi_expert_prepacked_int8(/*heterogeneous_k=*/true);
}

TEST(TestMatmulUnreorderRoute, HeterogeneousCkPrepackedInt8StaysOnCk) {
    run_multi_expert_prepacked_int8(
            /*heterogeneous_k=*/true, /*ck_prepacked=*/true);
}

void run_heterogeneous_s4_effective_group_case(
        bool compatible_effective_groups) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI custom kernel is unavailable";
    }

    constexpr int E = 2, rows = 4, N = 256, threads = 4;
    const std::vector<int> M(E, rows), K {64, 128}, N_vec(E, N);
    const std::vector<int> groups {4, compatible_effective_groups ? 8 : 4};
    std::vector<std::vector<bfloat16_t>> src(E), dst(E), ref(E);
    std::vector<std::vector<int8_t>> raw_weight(E);
    std::vector<std::vector<float>> src_scale(E), wei_scale(E);
    std::vector<aligned_ptr_t> packed(E);
    std::vector<const void *> src_ptr(E), weight_ptr(E);
    std::vector<void *> dst_ptr(E), ref_ptr(E);
    std::vector<matmul::matmul_params> params(E);

    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(rows) * K[e]);
        dst[e].resize(static_cast<size_t>(rows) * N);
        ref[e].resize(static_cast<size_t>(rows) * N);
        raw_weight[e].assign((static_cast<size_t>(N) * K[e] + 1) / 2, 0);
        src_scale[e].resize(rows);
        wei_scale[e].resize(static_cast<size_t>(groups[e]) * N);
        for (size_t i = 0; i < src[e].size(); ++i) {
            src[e][i] = bfloat16_t(0.025f
                    * static_cast<float>(
                            static_cast<int>((i + e * 3) % 19) - 9));
        }
        const size_t weight_elems = static_cast<size_t>(N) * K[e];
        for (size_t i = 0; i < weight_elems; ++i) {
            const uint8_t nibble
                    = static_cast<uint8_t>(
                              static_cast<int>((i * 7 + e) % 15) - 7)
                    & 0x0F;
            uint8_t &byte
                    = reinterpret_cast<uint8_t *>(raw_weight[e].data())[i >> 1];
            byte = (i & 1U) ? static_cast<uint8_t>((byte & 0x0F) | nibble << 4)
                            : static_cast<uint8_t>((byte & 0xF0) | nibble);
        }
        for (int g = 0; g < groups[e]; ++g) {
            for (int n = 0; n < N; ++n) {
                wei_scale[e][static_cast<size_t>(g) * N + n] = 0.02f
                        + 0.00025f * static_cast<float>((g + n + e) % 5);
            }
        }

        reorder::reorder_params_t prepack;
        prepack.is_prepack = true;
        prepack.num_threads = threads;
        prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
        prepack.prepack.wei_dtype = data_type_t::s4;
        prepack.prepack.src_dtype = data_type_t::s8;
        prepack.prepack.K = K[e];
        prepack.prepack.N = N;
        prepack.prepack.ldb = K[e];
        prepack.prepack.transposed = true;
        prepack.prepack.sym_group_size = K[e] / groups[e];
        const size_t packed_bytes = reorder::weight_prepack_size(prepack);
        if (packed_bytes == 0) {
            GTEST_SKIP() << "AOCL native S4 reorder is unavailable";
        }
        packed[e].reset(zendnnl_aligned_alloc(64, packed_bytes));
        ASSERT_NE(packed[e], nullptr);
        ASSERT_EQ(reorder::reorder_direct(
                          raw_weight[e].data(), packed[e].get(), prepack),
                status_t::success);

        auto &p = params[e];
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s4;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        p.weight_cache_type = 1;
        p.num_threads = threads;
        p.quant_params.src_scale.buff = src_scale[e].data();
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {rows, 1};
        p.quant_params.wei_scale.buff = wei_scale[e].data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {groups[e], N};
        src_ptr[e] = src[e].data();
        weight_ptr[e] = packed[e].get();
        dst_ptr[e] = dst[e].data();
        ref_ptr[e] = ref[e].data();
    }

    const std::vector<char> layout(E, 'r');
    const std::vector<bool> transA(E, false), transB(E, true);
    const std::vector<int> lda = K, ldb = K, ldc(E, N);
    const std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    const std::vector<const void *> bias(E, nullptr);
    const std::vector<bool> weights_const(E, true);

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    ::reset_grp_matmul_caches();
    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::group_matmul_direct(layout, transA, transB, M, N_vec, K,
                      alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst_ptr,
                      ldc, weights_const, params, /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
                    std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    if (compatible_effective_groups) {
        EXPECT_NE(std::string(mode).find("custom"), std::string::npos) << mode;
    } else {
        EXPECT_EQ(std::string(mode).find("custom"), std::string::npos) << mode;
    }

    ::reset_grp_matmul_caches();
    matmul_config_t::instance().set_custom_kernel_route(false);
    {
        mt::AlgoEnvGuard algo1(1);
        ASSERT_EQ(matmul::group_matmul_direct(layout, transA, transB, M, N_vec,
                          K, alpha, src_ptr, lda, weight_ptr, ldb, bias, beta,
                          ref_ptr, ldc, weights_const, params,
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success);
    }
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < dst[e].size(); ++i) {
            const float expected = static_cast<float>(ref[e][i]);
            const float actual = static_cast<float>(dst[e][i]);
            EXPECT_NEAR(actual, expected, 0.01f + 0.01f * std::fabs(expected))
                    << "expert=" << e << " i=" << i;
        }
    }
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute,
        HeterogeneousS4EffectiveGroupSizesFallBackAtFullN) {
    run_heterogeneous_s4_effective_group_case(
            /*compatible_effective_groups=*/false);
}

TEST(TestMatmulUnreorderRoute,
        HeterogeneousS4GroupCountsWithSameEffectiveGroupStayOnCk) {
    run_heterogeneous_s4_effective_group_case(
            /*compatible_effective_groups=*/true);
}

TEST(TestMatmulUnreorderRoute,
        MixedPlainAoclPackCannotMutateRawExpertBeforeFallback) {
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI custom kernel is unavailable";
    }

    constexpr int E = 2, Mv = 1, K = 64, N = 256, threads = 4;
    const int pack_nr = ck::plan_pack_nr(K, N);
    const size_t raw_bytes = static_cast<size_t>(N) * K;
    const size_t raw_capacity
            = std::max(raw_bytes, ck::packed_weight_size_int8(K, N, pack_nr));
    aligned_ptr_t raw_first(zendnnl_aligned_alloc(64, raw_capacity));
    ASSERT_NE(raw_first, nullptr);
    std::vector<int8_t> raw_second(raw_bytes);
    auto *raw_first_bytes = static_cast<int8_t *>(raw_first.get());
    for (size_t i = 0; i < raw_bytes; ++i) {
        raw_first_bytes[i] = static_cast<int8_t>(static_cast<int>(i % 23) - 11);
        raw_second[i]
                = static_cast<int8_t>(static_cast<int>((i * 5) % 23) - 11);
    }
    std::vector<int8_t> raw_first_snapshot(
            raw_first_bytes, raw_first_bytes + raw_bytes);

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = threads;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::s8;
    prepack.prepack.src_dtype = data_type_t::s8;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = K;
    prepack.prepack.transposed = true;
    prepack.prepack.sym_group_size = 0;
    const size_t blocked_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(blocked_bytes, 0u);
    aligned_ptr_t blocked(zendnnl_aligned_alloc(64, blocked_bytes));
    ASSERT_NE(blocked, nullptr);
    ASSERT_EQ(
            reorder::reorder_direct(raw_second.data(), blocked.get(), prepack),
            status_t::success);

    std::vector<std::vector<int8_t>> src(
            E, std::vector<int8_t>(static_cast<size_t>(Mv) * K));
    std::vector<std::vector<bfloat16_t>> dst(
            E, std::vector<bfloat16_t>(static_cast<size_t>(Mv) * N));
    std::vector<std::vector<float>> wei_scale(E, std::vector<float>(N));
    std::vector<float> src_scale(E, 0.02f);
    std::vector<matmul::matmul_params> params(E);
    std::vector<const void *> src_ptr(E),
            weight_ptr {
                    raw_first.get(),
                    blocked.get(),
            };
    std::vector<void *> dst_ptr(E);
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < src[e].size(); ++i) {
            src[e][i] = static_cast<int8_t>(static_cast<int>((i + e) % 17) - 8);
        }
        for (int n = 0; n < N; ++n) {
            wei_scale[e][n]
                    = 0.0125f + 0.00025f * static_cast<float>((n + e) % 7);
        }
        auto &p = params[e];
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.mem_format_b = e == 0 ? 'n' : 'r';
        p.lowoha_algo = e == 0 ? matmul_algo_t::none
                               : matmul_algo_t::aocl_dlp_blocked;
        p.weight_cache_type = 2;
        p.wei_buffer_capacity_bytes = e == 0 ? raw_capacity : 0;
        p.num_threads = threads;
        p.quant_params.src_scale.buff = &src_scale[e];
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {1, 1};
        p.quant_params.wei_scale.buff = wei_scale[e].data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {N};
        src_ptr[e] = src[e].data();
        dst_ptr[e] = dst[e].data();
    }

    const std::vector<char> layout(E, 'r');
    const std::vector<bool> transA(E, false), transB(E, true);
    const std::vector<int> M(E, Mv), N_vec(E, N), K_vec(E, K), lda(E, K),
            ldb(E, K), ldc(E, N);
    const std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    const std::vector<const void *> bias(E, nullptr);
    const std::vector<bool> weights_const(E, true);

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(2);
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride custom_on(true);
    mt::CustomKernelInt8Override int8_on(true);
    mt::NTileStrategyOverride rounds(2);
    ::reset_grp_matmul_caches();
    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::group_matmul_direct(layout, transA, transB, M, N_vec,
                      K_vec, alpha, src_ptr, lda, weight_ptr, ldb, bias, beta,
                      dst_ptr, ldc, weights_const, params,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
            status_t::success);
    EXPECT_EQ(
            std::memcmp(raw_first.get(), raw_first_snapshot.data(), raw_bytes),
            0);
    EXPECT_FALSE(ck::any_weight_packed_in_place(weight_ptr, M));
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
                    std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(std::string(mode).find("custom"), std::string::npos) << mode;
    ::reset_grp_matmul_caches();
}

TEST(TestMatmulUnreorderRoute, SimulatedW4A8SkipsAoclBlockedPlainConversion) {
    constexpr int K = 64, N = 256;
    int8_t blocked_dummy = 0;
    std::vector<const void *> weight {&blocked_dummy};
    std::vector<matmul::matmul_params> params(1);
    params[0].dtypes.src = data_type_t::bf16;
    params[0].dtypes.wei = data_type_t::s4;
    params[0].dtypes.dst = data_type_t::bf16;
    params[0].dtypes.compute = data_type_t::s8;
    params[0].dynamic_quant = true;
    params[0].mem_format_b = 'r';
    static float src_scale = 1.0f, wei_scale = 1.0f;
    params[0].quant_params.src_scale.buff = &src_scale;
    params[0].quant_params.src_scale.dt = data_type_t::f32;
    params[0].quant_params.src_scale.dims = {1, 1};
    params[0].quant_params.wei_scale.buff = &wei_scale;
    params[0].quant_params.wei_scale.dt = data_type_t::f32;
    params[0].quant_params.wei_scale.dims = {K / 16, N};
    std::vector<void *> converted;
    bool any_w4a8 = true;
    ASSERT_TRUE(matmul::w4a8_populate_plain_s8_cache(weight,
            /*K=*/ {K}, /*N=*/ {N}, /*ldb=*/ {K}, /*transB=*/ {true}, params,
            /*is_weights_const=*/ {true}, /*num_ops=*/1, converted, any_w4a8));
    ASSERT_EQ(converted.size(), 1u);
    EXPECT_EQ(converted[0], nullptr);
    EXPECT_FALSE(any_w4a8);
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
    if (!ck::bf16_isa_supported()) {
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

    mt::ExecutionPathCaptureGuard mode_capture;
    ASSERT_EQ(matmul::group_matmul_direct(layout, trans_a, trans_b, ms, ns, ks,
                      alpha, src_ptr, lda, weight_ptr, ldb, bias, beta, dst_ptr,
                      ldc, weights_const, params, /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);
    const char *mode
            = matmul::test_api::s_last_group_matmul_direct_execution_path.load(
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

// ── Single-expert policy (see single_expert/ for the policy tests) ──

// One raw narrow-N weight goes prompt -> decode -> prompt through
// `matmul_direct`.  At 32 threads it is below decode's N-tile count (only
// FORCE_CK covers it); at 8 threads its 40 NR32 tiles pass the gate, so the
// phase knobs cover it too.  With the single-expert prompt tile knob
// on, both phases route to the custom kernel, so a WEIGHT_CACHE=2 in-place
// pack is the custom-kernel layout and every later call reads it; the buffer
// never holds an AOCL layout.  With the knob off, the weight stays raw.
void run_narrow_phase_switch(int weight_cache_mode, bool with_bias, int knob,
        int algo_pin = 0, uint32_t lru_capacity = 0, int force_ck = 0,
        int decode_knob = -1, int threads = 32) {
    if (decode_knob < 0) decode_knob = knob;
    constexpr int N = 1280, K = 768;
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
    mt::SingleExpertCkPolicy knob_guard(knob, decode_knob, force_ck);
    matmul::clear_aocl_ck_weight_cache();
    ::reset_grp_matmul_caches();
    namespace pt = zendnnl::lowoha::matmul::prompt_tile;
    const long served0 = pt::test_api::s_calls_served.load();

    // A phase knob covers its phase; below decode's N-tile count only under
    // FORCE_CK.
    const bool n_ok = force_ck != 0
            || matmul::has_enough_single_expert_tiles(N, K, threads);
    const bool prompt_cov = knob != 0 && n_ok;
    const bool decode_cov = decode_knob != 0 && n_ok;
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
        {
            mt::ExecutionPathCaptureGuard mode_capture;
            ASSERT_EQ(matmul::matmul_direct('r', false, /*transB=*/true, M, N,
                              K, 1.0f, src.data(), K, weight.data(), K,
                              with_bias ? bias.data() : nullptr, 0.0f,
                              dst.data(), N, true, batch, params),
                    status_t::success);
            const char *mode
                    = matmul::test_api::
                              s_last_group_matmul_direct_execution_path.load(
                                      std::memory_order_relaxed);
            if (M <= 4 && decode_cov) {
                ASSERT_NE(mode, nullptr);
                EXPECT_NE(std::string(mode).find("custom"), std::string::npos)
                        << "a covered decode runs the custom kernel: " << mode;
            }
        }
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
    if (prompt_cov && decode_cov) {
        EXPECT_GT(pt::test_api::s_calls_served.load() - served0, 0)
                << "prompts must run the prompt tile";
        EXPECT_TRUE(!mutated || ck::weight_packed_in_place(weight.data()))
                << "an in-place image must be the custom-kernel pack";
    } else if (prompt_cov) {
        EXPECT_GT(pt::test_api::s_calls_served.load() - served0, 0)
                << "prompts must run the prompt tile";
    } else if (!decode_cov && !n_ok) {
        EXPECT_FALSE(mutated) << "policy off keeps the raw weight";
    }
    ::reset_grp_matmul_caches();
    matmul::clear_aocl_ck_weight_cache();
}

TEST(TestMatmulUnreorderRoute, NarrowWeightPromptDecodeShareCkPack) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    for (const int knob : {1, 0})
        for (const int wc : {1, 2})
            for (const bool bias : {false, true}) {
                SCOPED_TRACE("knob=" + std::to_string(knob) + " wc="
                        + std::to_string(wc) + " bias=" + std::to_string(bias));
                run_narrow_phase_switch(wc, bias, knob, 0, 0, 0, -1,
                        /*threads=*/knob != 0 ? 8 : 32);
            }
    for (const int wc : {1, 2})
        for (const bool bias : {false, true}) {
            SCOPED_TRACE("force_ck wc=" + std::to_string(wc)
                    + " bias=" + std::to_string(bias));
            run_narrow_phase_switch(wc, bias, /*knob=*/1, 0, 0, /*force_ck=*/1);
        }
    // One phase on the custom kernel, the other on AOCL.
    for (const auto &[pk, dk] : {std::pair {1, 0}, std::pair {0, 1}})
        for (const int wc : {1, 2})
            for (const bool bias : {false, true}) {
                SCOPED_TRACE("prompt=" + std::to_string(pk) + " decode="
                        + std::to_string(dk) + " wc=" + std::to_string(wc)
                        + " bias=" + std::to_string(bias));
                run_narrow_phase_switch(
                        wc, bias, pk, 0, 0, 0, dk, /*threads=*/8);
            }
}

// The WEIGHT_CACHE=2 settings that turn AUTO mixed in-place mode off (an
// ALGO 3 pin, a finite LRU capacity) let the prompt's pack go in place; the
// decode must then read it through the custom kernel.
TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackWithoutMixedMode) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    for (const bool bias : {false, true}) {
        SCOPED_TRACE("algo3 pin, bias=" + std::to_string(bias));
        run_narrow_phase_switch(2, bias, 1, /*algo_pin=*/3, 0, 0, -1, 8);
    }
    for (const bool bias : {false, true}) {
        SCOPED_TRACE("LRU capacity 64, bias=" + std::to_string(bias));
        run_narrow_phase_switch(2, bias, 1, 0, /*lru_capacity=*/64, 0, -1, 8);
    }
}

// With both phases covered (here by FORCE_CK), WEIGHT_CACHE=2 may pack a raw
// BF16 single-expert weight in place.  A later call on that buffer that the route
// does not take (here an F32 destination) cannot read the image and must fail
// closed, not compute on packed bytes.
TEST(TestMatmulUnreorderRoute,
        NonRoutedCallOnSingleExpertInPlaceImageFailsClosed) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    constexpr int N = 1280, K = 768, M = 256, threads = 32;
    std::vector<bfloat16_t> weight(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight[static_cast<size_t>(n) * K + k]
                    = bfloat16_t(logical_weight(k, n));
    std::vector<bfloat16_t> src(static_cast<size_t>(M) * K, bfloat16_t(0.25f));

    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(2);
    mt::AlgoEnvGuard algo(3);
    mt::CustomKernelOverride custom_on(true);
    mt::SingleExpertCkPolicy force(1, 1, 1);
    matmul::clear_aocl_ck_weight_cache();
    ::reset_grp_matmul_caches();

    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    params.num_threads = threads;
    matmul::matmul_batch_params_t batch;
    std::vector<bfloat16_t> dst(static_cast<size_t>(M) * N);
    ASSERT_EQ(matmul::matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      src.data(), K, weight.data(), K, nullptr, 0.0f,
                      dst.data(), N, true, batch, params),
            status_t::success);
    if (!ck::weight_packed_in_place(weight.data())) {
        ::reset_grp_matmul_caches();
        GTEST_SKIP() << "the pack did not go in place on this configuration";
    }
    params.dtypes.dst = data_type_t::f32;
    const float canary = -7.0f;
    std::vector<float> dst_f32(static_cast<size_t>(M) * N, canary);
    EXPECT_NE(matmul::matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      src.data(), K, weight.data(), K, nullptr, 0.0f,
                      dst_f32.data(), N, true, batch, params),
            status_t::success);
    ::reset_grp_matmul_caches();
    matmul::clear_aocl_ck_weight_cache();
}

// CROSS_WARM and PREPACK are read once per process, so each runs in a child.
void expect_narrow_phase_switch_in_child(const char *env, const char *value) {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    setenv(env, value, /*overwrite=*/1);
    EXPECT_EXIT(
            {
                run_narrow_phase_switch(
                        2, /*with_bias=*/false, 1, 0, 0, 0, -1, 8);
                run_narrow_phase_switch(
                        2, /*with_bias=*/true, 1, 0, 0, 0, -1, 8);
                run_narrow_phase_switch(2, /*with_bias=*/false, 1, 0, 0, 1);
                run_narrow_phase_switch(
                        2, /*with_bias=*/false, 1, 0, 0, 0, 0, 8);
                run_narrow_phase_switch(
                        2, /*with_bias=*/false, 0, 0, 0, 0, 1, 8);
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    unsetenv(env);
}

// With PREPACK off, AUTO mixed in-place mode cannot start and the first call
// downgrades the process to WEIGHT_CACHE=1.  A prompt-only layer whose decode
// leaves ALGO 3 (a decode pin, or Rule 0.45 off) must then find its weight
// unpacked: the first prompt may not pack it in place.
TEST(TestMatmulUnreorderRoute,
        NarrowPromptOnlyLayerWithPrepackOffKeepsRawWeight) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    setenv("ZENDNNL_GRP_MATMUL_PREPACK", "0", /*overwrite=*/1);
    EXPECT_EXIT(
            {
                {
                    mt::AutoDecodeAlgoOverride decode_pin(1);
                    run_narrow_phase_switch(
                            2, /*with_bias=*/false, 1, 0, 0, 0, 0, 8);
                }
                {
                    mt::DenseDecodeNTileOverride rule045_off(0);
                    run_narrow_phase_switch(
                            2, /*with_bias=*/true, 1, 0, 0, 0, 0, 8);
                }
                std::exit(::testing::Test::HasFailure() ? 1 : 0);
            },
            ::testing::ExitedWithCode(0), "");
    unsetenv("ZENDNNL_GRP_MATMUL_PREPACK");
}

TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackCrossWarmOff) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    expect_narrow_phase_switch_in_child("ZENDNNL_GRP_MATMUL_CROSS_WARM", "0");
}

TEST(TestMatmulUnreorderRoute, NarrowWeightInPlaceCkPackPrepackOff) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    expect_narrow_phase_switch_in_child("ZENDNNL_GRP_MATMUL_PREPACK", "0");
}

// The prompt knob needs decode's N-tile count, and `decode_d`
// (N_TILE_STRATEGY=1) never reaches the prompt tile executor: a narrow BF16
// layer under the prompt knob keeps the standard MatMul path either way, and
// FORCE_CK routes it either way.
TEST(TestMatmulUnreorderRoute, NarrowBf16PromptDecodeDKeepsStandardPath) {
    if (!ck::bf16_isa_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 custom kernel is unavailable";
    }
    RouteConfigGuard route_config;
    WeightCacheGuard cache_mode(1);
    mt::AlgoEnvGuard algo(0);
    mt::CustomKernelOverride custom_on(true);
    matmul::matmul_params params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.wei = data_type_t::bf16;
    params.dtypes.dst = data_type_t::bf16;
    const auto routable = [&](int M) {
        return matmul::single_matmul_group_routable('r', M, 1280, 768, 32, 1,
                false, true, 1.0f, 0.0f, nullptr, 768, true, params);
    };
    for (const int strategy : {2, 1}) {
        SCOPED_TRACE(strategy);
        mt::NTileStrategyOverride st(strategy);
        {
            mt::SingleExpertCkPolicy prompt_only(1, 0, 0);
            EXPECT_FALSE(routable(2048));
            EXPECT_FALSE(routable(4));
            EXPECT_EQ(matmul::grp_matmul_single_expert_ck(params, 768, 1280, 8)
                              .prompt,
                    strategy != 1)
                    << "with the N-tile count, only decode_d drops the prompt";
        }
        mt::SingleExpertCkPolicy force(1, 1, 1);
        EXPECT_TRUE(routable(2048));
        EXPECT_TRUE(routable(4));
    }
}

} // namespace
