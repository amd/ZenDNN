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

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
namespace matmul = zendnnl::lowoha::matmul;
namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace reorder = zendnnl::lowoha::reorder;

struct aligned_deleter {
    void operator()(void *ptr) const noexcept {
        if (ptr != nullptr) zendnnl_aligned_free(ptr);
    }
};
using aligned_ptr_t = std::unique_ptr<void, aligned_deleter>;

class CacheCapacityGuard {
public:
    CacheCapacityGuard()
        : config_(zendnnl::common::matmul_config_t::instance())
        , previous_(config_.get_lru_cache_capacity()) {
        config_.set_lru_cache_capacity(std::numeric_limits<uint32_t>::max());
    }
    ~CacheCapacityGuard() { config_.set_lru_cache_capacity(previous_); }

private:
    zendnnl::common::matmul_config_t &config_;
    uint32_t previous_;
};

class WeightCacheModeGuard {
public:
    explicit WeightCacheModeGuard(int32_t mode)
        : config_(zendnnl::common::matmul_config_t::instance())
        , previous_(config_.get_weight_cache()) {
        config_.set_weight_cache(mode);
    }
    ~WeightCacheModeGuard() { config_.set_weight_cache(previous_); }

private:
    zendnnl::common::matmul_config_t &config_;
    int32_t previous_;
};

class DlpMHintGuard {
public:
    explicit DlpMHintGuard(int64_t hint)
        : config_(zendnnl::common::matmul_config_t::instance())
        , previous_(config_.get_dlp_m_hint()) {
        config_.set_dlp_m_hint(hint);
    }
    ~DlpMHintGuard() { config_.set_dlp_m_hint(previous_); }

private:
    zendnnl::common::matmul_config_t &config_;
    int64_t previous_;
};

void run_derived_ck_cache_case(
        bool transposed, int K = 64, int ldb_padding = 0) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    GTEST_SKIP() << "AOCL-DLP is unavailable";
#else
    constexpr int N = 128;
    const int ldb = (transposed ? K : N) + ldb_padding;
    const int rows = transposed ? N : K;

    std::vector<bfloat16_t> original(static_cast<size_t>(rows) * ldb);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            original[transposed ? static_cast<size_t>(n) * ldb + k
                                : static_cast<size_t>(k) * ldb + n]
                    = bfloat16_t(0.01f
                            * static_cast<float>((k * 17 + n * 7) % 19 - 9));

    reorder::reorder_params_t prepack;
    prepack.is_prepack = true;
    prepack.num_threads = 4;
    prepack.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    prepack.prepack.wei_dtype = data_type_t::bf16;
    prepack.prepack.src_dtype = data_type_t::bf16;
    prepack.prepack.K = K;
    prepack.prepack.N = N;
    prepack.prepack.ldb = ldb;
    prepack.prepack.transposed = transposed;

    const size_t packed_bytes = reorder::weight_prepack_size(prepack);
    ASSERT_GT(packed_bytes, 0u);
    aligned_ptr_t packed(zendnnl_aligned_alloc(64, packed_bytes));
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(reorder::reorder_direct(original.data(), packed.get(), prepack),
            status_t::success);

    constexpr int pack_nr = 32;
    const size_t ck_bytes = ck::packed_weight_size_bf16(K, N, pack_nr);
    ASSERT_GT(ck_bytes, 0u);
    if ((K & 1) != 0 && ldb_padding == 0) {
        EXPECT_GT(ck_bytes, original.size() * sizeof(bfloat16_t));
    }
    aligned_ptr_t expected_ck(zendnnl_aligned_alloc(64, ck_bytes));
    ASSERT_NE(expected_ck, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_bf16(original.data(), K, N, ldb, pack_nr,
                      transposed, /*interleave_split_halves=*/false,
                      expected_ck.get()),
            status_t::success);

    CacheCapacityGuard capacity_guard;
    matmul::clear_aocl_ck_weight_cache();

    std::shared_ptr<void> ck_first;
    bool hit = true;
    ASSERT_EQ(matmul::get_or_cache_aocl_ck_weight(packed.get(), K, N, ldb,
                      transposed, /*num_threads=*/4, pack_nr,
                      /*interleave_split_halves=*/false, ck_first, &hit),
            status_t::success);
    EXPECT_FALSE(hit);
    ASSERT_NE(ck_first, nullptr);
    EXPECT_EQ(std::memcmp(ck_first.get(), expected_ck.get(), ck_bytes), 0);

    std::shared_ptr<void> ck_second;
    ASSERT_EQ(matmul::get_or_cache_aocl_ck_weight(packed.get(), K, N, ldb,
                      transposed, /*num_threads=*/4, pack_nr,
                      /*interleave_split_halves=*/false, ck_second, &hit),
            status_t::success);
    EXPECT_TRUE(hit);
    EXPECT_EQ(ck_first.get(), ck_second.get());

    ck_second.reset();
    matmul::clear_grp_matmul_weight_caches();
    ASSERT_NE(ck_first, nullptr)
            << "an outstanding lease must survive explicit cache clear";
    EXPECT_EQ(std::memcmp(ck_first.get(), expected_ck.get(), ck_bytes), 0);

    std::shared_ptr<void> ck_after_clear;
    ASSERT_EQ(matmul::get_or_cache_aocl_ck_weight(packed.get(), K, N, ldb,
                      transposed, /*num_threads=*/4, pack_nr,
                      /*interleave_split_halves=*/false, ck_after_clear, &hit),
            status_t::success);
    EXPECT_FALSE(hit);
    EXPECT_EQ(
            std::memcmp(ck_after_clear.get(), expected_ck.get(), ck_bytes), 0);

    ck_first.reset();
    ck_after_clear.reset();
    matmul::clear_aocl_matmul_weight_caches();
#endif
}

TEST(TestAoclDerivedCkWeightCache, NonTransposedRoundTripAndHit) {
    run_derived_ck_cache_case(false);
}

TEST(TestAoclDerivedCkWeightCache, TransposedRoundTripAndHit) {
    run_derived_ck_cache_case(true);
}

TEST(TestAoclDerivedCkWeightCache, OddKUsesCkSizedArena) {
    run_derived_ck_cache_case(/*transposed=*/true, /*K=*/65);
}

TEST(TestAoclDerivedCkWeightCache, DisabledWeightCacheDeclinesPersistentEntry) {
    CacheCapacityGuard capacity_guard;
    WeightCacheModeGuard cache_off(0);
    int dummy_packed = 0;
    std::shared_ptr<void> ck_weight;
    bool hit = true;
    EXPECT_EQ(matmul::get_or_cache_aocl_ck_weight(&dummy_packed, /*K=*/64,
                      /*N=*/128, /*ldb=*/128, /*transB=*/false,
                      /*num_threads=*/4, /*pack_nr=*/32,
                      /*interleave_split_halves=*/false, ck_weight, &hit),
            status_t::unimplemented);
    EXPECT_EQ(ck_weight, nullptr);
    EXPECT_FALSE(hit);
}

TEST(TestAoclDerivedCkWeightCache,
        PerCallDisabledCacheDeclinesPersistentEntry) {
    CacheCapacityGuard capacity_guard;
    WeightCacheModeGuard global_cache_on(1);
    int dummy_packed = 0;
    std::shared_ptr<void> ck_weight;
    bool hit = true;
    EXPECT_EQ(matmul::get_or_cache_aocl_ck_weight(&dummy_packed, /*K=*/64,
                      /*N=*/128, /*ldb=*/128, /*transB=*/false,
                      /*num_threads=*/4, /*pack_nr=*/32,
                      /*interleave_split_halves=*/false, ck_weight, &hit,
                      /*weight_cache_mode=*/0),
            status_t::unimplemented);
    EXPECT_EQ(ck_weight, nullptr);
    EXPECT_FALSE(hit);
}

TEST(TestAoclDerivedCkWeightCache,
        MHintWithoutOriginalThreadProvenanceDeclines) {
    CacheCapacityGuard capacity_guard;
    WeightCacheModeGuard cache_on(1);
    DlpMHintGuard m_hint(32);
    int dummy_packed = 0;
    std::shared_ptr<void> ck_weight;
    bool hit = true;
    EXPECT_EQ(matmul::get_or_cache_aocl_ck_weight(&dummy_packed, /*K=*/64,
                      /*N=*/128, /*ldb=*/128, /*transB=*/false,
                      /*num_threads=*/8, /*pack_nr=*/32,
                      /*interleave_split_halves=*/false, ck_weight, &hit),
            status_t::unimplemented);
    EXPECT_EQ(ck_weight, nullptr);
    EXPECT_FALSE(hit);
}

} // namespace
