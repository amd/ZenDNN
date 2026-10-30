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

/// Which MoE layers AUTO keeps on ALGO 3 in both phases
/// (`grp_matmul_auto_algo3_family`), with the INT8 / W4A8 gates, the dtype
/// knob and the WEIGHT_CACHE=2 guard that read it.

#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "auto_probe_test_utils.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/multi_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/selection/algo_select.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
using auto_probe_test_utils::AutoProbeShape;
using auto_probe_test_utils::build_auto_probe;
using zendnnl::common::data_type_t;

enum class quant_t { dq_int8, s8_src, w4a8 };

const char *name_of(quant_t q) {
    switch (q) {
        case quant_t::dq_int8: return "dq_int8";
        case quant_t::s8_src: return "s8_src";
        default: return "w4a8";
    }
}

/// An MoE layer of `num_ops` experts, the first `active` of them firing.
/// INT8 has per-channel weight scales; W4A8 has groups of 64 along K.  The
/// BF16 source is quantized per token at run time, or (`s8_src`) arrives
/// already quantized, as routed MoE's generic path hands it over.
AutoProbeShape quant_layer(quant_t q, int m, int num_ops, int active) {
    static std::vector<float> scale(1 << 16, 1.0f / 127.0f);
    AutoProbeShape s = build_auto_probe(m, /*K=*/2048, /*N=*/1536, num_ops,
            /*num_threads=*/32);
    for (int i = active; i < num_ops; ++i)
        s.M[i] = 0;
    for (size_t i = 0; i < s.params.size(); ++i) {
        auto &p = s.params[i];
        const int64_t rows = s.M[i] > 0 ? s.M[i] : 1;
        p.dtypes.src
                = q == quant_t::s8_src ? data_type_t::s8 : data_type_t::bf16;
        p.dtypes.wei = q == quant_t::w4a8 ? data_type_t::s4 : data_type_t::s8;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = q != quant_t::s8_src;
        p.quant_params.src_scale.dims = {rows, 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.buff = scale.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = scale.data();
        p.quant_params.wei_scale.dims = q == quant_t::w4a8
                ? std::vector<int64_t> {s.K[i] / 64, s.N[i]}
                : std::vector<int64_t> {1, s.N[i]};
    }
    return s;
}

std::string pick_reason(const AutoProbeShape &s, int *algo = nullptr) {
    mm::GroupMatmulAlgoTrace trace;
    const int a = mm::group_matmul_algo_select(
            s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace);
    if (algo != nullptr) *algo = a;
    return trace.reason != nullptr ? trace.reason : "<none>";
}

bool algo3_call(const AutoProbeShape &s) {
    return mm::grp_matmul_auto_algo3_call(s.M, s.N, s.K, s.params);
}

class MoeAlgo3Quant : public ::testing::Test {
protected:
    void SetUp() override {
        if (!mm::custom_kernel::avx512vnni_available())
            GTEST_SKIP() << "INT8 / W4A8 custom kernels need AVX-512 VNNI";
        reset_grp_matmul_caches();
    }
    moe_test_utils::AlgoEnvGuard algo_ {0};
    moe_test_utils::CustomKernelOverride ck_ {true};
    moe_test_utils::CustomKernelInt8Override ck_int8_ {true};
};

// Prompt and decode of every INT8 / W4A8 form run ALGO 3, including a
// Mixtral-class decode (8 experts, 2 firing) the occupancy rule would send to
// ALGO 1.
TEST_F(MoeAlgo3Quant, BothPhasesRunAlgo3) {
    for (quant_t q : {quant_t::dq_int8, quant_t::s8_src, quant_t::w4a8}) {
        SCOPED_TRACE(name_of(q));
        for (const auto &c : {std::make_pair(512, 8), std::make_pair(4, 8),
                     std::make_pair(512, 2), std::make_pair(4, 2)}) {
            SCOPED_TRACE("M=" + std::to_string(c.first)
                    + " active=" + std::to_string(c.second));
            const AutoProbeShape s = quant_layer(q, c.first, 8, c.second);
            EXPECT_TRUE(algo3_call(s));
            // Its pack is out of place, so the dispatcher grants it no
            // custom-kernel layout licence.
            EXPECT_EQ(mm::grp_matmul_auto_algo3_family(s.M, s.N, s.K, s.params),
                    mm::grp_matmul_algo3_family::quant_family);
            int algo = 0;
            EXPECT_EQ(pick_reason(s, &algo), "auto_algo3");
            EXPECT_EQ(algo, 3);
        }
    }
}

// An explicit `ZENDNNL_GRP_MATMUL_ALGO=0` keeps the shape rules: prompt on
// Rule 0.7, a Mixtral-class decode on Rule 0.5.
TEST_F(MoeAlgo3Quant, ExplicitAlgoZeroKeepsTheShapeRules) {
    moe_test_utils::AlgoLegacyOverride legacy(true);
    for (quant_t q : {quant_t::dq_int8, quant_t::s8_src, quant_t::w4a8}) {
        SCOPED_TRACE(name_of(q));
        EXPECT_FALSE(algo3_call(quant_layer(q, 512, 8, 8)));
        EXPECT_EQ(pick_reason(quant_layer(q, 512, 8, 8)),
                "auto_rule07_prompt_seq");
        EXPECT_EQ(
                pick_reason(quant_layer(q, 4, 8, 2)), "auto_rule05_few_active");
    }
}

// A layer the custom kernel cannot serve in both phases keeps the shape
// rules, so its prompt stays on AOCL-DLP.
TEST_F(MoeAlgo3Quant, LayersOutsideTheKernelKeepTheShapeRules) {
    const auto refused = [](const AutoProbeShape &s, const char *why) {
        SCOPED_TRACE(why);
        EXPECT_FALSE(algo3_call(s));
        EXPECT_NE(pick_reason(s), "auto_algo3");
    };
    {
        moe_test_utils::CustomKernelInt8Override int8_off(false);
        refused(quant_layer(quant_t::dq_int8, 512, 8, 8), "INT8 kernel off");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &k : s.K)
            k = 2046;
        refused(s, "K not a multiple of 4");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (size_t i = 0; i < s.params.size(); ++i)
            s.params[i].quant_params.wei_scale.dims = {s.K[i] / 128, s.N[i]};
        refused(s, "per-group INT8 weight scale");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        s.params[5].quant_params.wei_scale.dims = {s.K[5] / 128, s.N[5]};
        refused(s, "one per-group INT8 expert among per-channel ones");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        s.K[6] = 2046;
        refused(s, "one expert with K not a multiple of 4");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::w4a8, 512, 8, 8);
        for (size_t i = 0; i < s.params.size(); ++i)
            s.params[i].quant_params.src_scale.dims = {s.M[i], s.K[i] / 64};
        refused(s, "per-group W4A8 source scale");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &p : s.params) {
            p.mem_format_b = 'r';
            p.lowoha_algo = zendnnl::common::matmul_algo_t::aocl_dlp_blocked;
        }
        refused(s, "AOCL-prepacked INT8 weight");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        const AutoProbeShape w4 = quant_layer(quant_t::w4a8, 512, 8, 8);
        s.params[1] = w4.params[1];
        refused(s, "INT8 and W4A8 experts in one call");
    }
    {
        static float dst_scale = 2.0f;
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &p : s.params) {
            p.quant_params.dst_scale.buff = &dst_scale;
            p.quant_params.dst_scale.dt = data_type_t::f32;
            p.quant_params.dst_scale.dims = {1, 1};
        }
        refused(s, "a destination scale");
        AutoProbeShape one = quant_layer(quant_t::dq_int8, 512, 8, 8);
        one.params[7].quant_params.dst_scale
                = s.params[7].quant_params.dst_scale;
        refused(one, "one expert with a destination scale");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::w4a8, 512, 8, 8);
        auto &d = s.params[3].quant_params.wei_scale.dims;
        ASSERT_EQ(d.size(), 2u);
        d[0] *= 2;
        refused(s, "W4A8 experts with different group sizes");
    }
    refused(quant_layer(quant_t::dq_int8, 512, 1, 1), "a single expert");
    {
        moe_test_utils::CustomKernelOverride ck_off(false);
        refused(quant_layer(quant_t::w4a8, 512, 8, 8), "custom kernel off");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &p : s.params)
            p.postop_.resize(1);
        refused(s, "a post-op");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &p : s.params) {
            p.dtypes.src = data_type_t::u8;
            p.dtypes.compute = data_type_t::none;
            p.dynamic_quant = false;
        }
        refused(s, "static U8xS8");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &p : s.params)
            p.dtypes.compute = data_type_t::u8;
        refused(s, "asymmetric DQ-INT8");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &n : s.N)
            n = 1528;
        refused(s, "N not a multiple of the pack width");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::dq_int8, 512, 8, 8);
        for (auto &k : s.K)
            k = (mm::custom_kernel::kMaxSafeInt8AccumK / 4 + 1) * 4;
        refused(s, "K past the INT8 accumulation depth");
    }
    {
        AutoProbeShape s = quant_layer(quant_t::w4a8, 512, 8, 8);
        for (size_t i = 0; i < s.params.size(); ++i)
            s.params[i].quant_params.wei_scale.dims = {s.K[i] / 4, s.N[i]};
        refused(s, "W4A8 group of 4 (not a multiple of 8)");
    }
    static int32_t zp[1] = {0};
    for (quant_t q : {quant_t::dq_int8, quant_t::w4a8}) {
        SCOPED_TRACE(name_of(q));
        {
            AutoProbeShape s = quant_layer(q, 512, 8, 8);
            for (auto &p : s.params)
                p.quant_params.wei_zp.buff = zp;
            refused(s, "a weight zero point");
        }
        {
            AutoProbeShape s = quant_layer(q, 512, 8, 8);
            for (auto &p : s.params)
                p.quant_params.src_zp.buff = zp;
            refused(s, "a source zero point");
        }
        {
            AutoProbeShape s = quant_layer(q, 512, 8, 8);
            for (auto &p : s.params) {
                p.mem_format_b = 'r';
                p.lowoha_algo
                        = zendnnl::common::matmul_algo_t::aocl_dlp_blocked;
            }
            refused(s, "an AOCL-prepacked weight");
        }
    }
}

// One firing expert of a declared pool stays with its layer.
TEST_F(MoeAlgo3Quant, OneActiveExpertOfAPoolStaysOnAlgo3) {
    AutoProbeShape s = quant_layer(quant_t::s8_src, 4, 1, 1);
    for (auto &p : s.params) {
        p.active_matmul = 1;
        p.total_matmul = 8;
    }
    EXPECT_TRUE(algo3_call(s));
    EXPECT_EQ(pick_reason(s), "auto_algo3");
}

// The group DQ pre-pass rewrites a BF16 dynamically quantized source before
// the dispatcher and the executor see it, while the fused-MoE arena is chosen
// from the original form: both forms must get the same verdict.
TEST_F(MoeAlgo3Quant, PreAndPostDqFormsAgree) {
    using fam = mm::grp_matmul_algo3_family;
    const auto family = [](const AutoProbeShape &s) {
        return mm::grp_matmul_auto_algo3_family(s.M, s.N, s.K, s.params);
    };
    // Symmetric INT8: BF16 + dynamic_quant, then S8 with dynamic_quant off.
    EXPECT_EQ(family(quant_layer(quant_t::dq_int8, 512, 8, 8)),
            fam::quant_family);
    EXPECT_EQ(
            family(quant_layer(quant_t::s8_src, 512, 8, 8)), fam::quant_family);
    // W4A8: BF16 + dynamic_quant, then S8.
    AutoProbeShape w4 = quant_layer(quant_t::w4a8, 512, 8, 8);
    EXPECT_EQ(family(w4), fam::quant_family);
    for (auto &p : w4.params) {
        p.dtypes.src = data_type_t::s8;
        p.dynamic_quant = false;
    }
    EXPECT_EQ(family(w4), fam::quant_family);
    // Asymmetric INT8: BF16 + dynamic_quant + u8 compute, then a U8 source
    // with its zero point.  Both keep the shape rules.
    static int32_t zp[1 << 10] = {};
    AutoProbeShape asym = quant_layer(quant_t::dq_int8, 512, 8, 8);
    for (auto &p : asym.params)
        p.dtypes.compute = data_type_t::u8;
    EXPECT_EQ(family(asym), fam::none);
    for (auto &p : asym.params) {
        p.dtypes.src = data_type_t::u8;
        p.dynamic_quant = false;
        p.quant_params.src_zp.buff = zp;
    }
    EXPECT_EQ(family(asym), fam::none);
}

// `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3` picks the dtypes; its default under
// an explicit `ZENDNNL_GRP_MATMUL_ALGO=0` is BF16 and FP16.
TEST_F(MoeAlgo3Quant, DtypeKnobSelectsTheFamilies) {
    using fam = mm::grp_matmul_algo3_family;
    struct Knob {
        int prev;
        explicit Knob(int v) {
            prev = mm::test_api::s_grp_matmul_multi_expert_algo3_override
                           .exchange(v, std::memory_order_relaxed);
        }
        ~Knob() {
            mm::test_api::s_grp_matmul_multi_expert_algo3_override.store(
                    prev, std::memory_order_relaxed);
        }
    };
    const auto family = [](const AutoProbeShape &s) {
        return mm::grp_matmul_auto_algo3_family(s.M, s.N, s.K, s.params);
    };
    const AutoProbeShape bf16 = build_auto_probe(512, 2048, 1536, 8, 32);
    const AutoProbeShape int8 = quant_layer(quant_t::s8_src, 512, 8, 8);
    const AutoProbeShape w4a8 = quant_layer(quant_t::w4a8, 512, 8, 8);
    {
        Knob k(mm::kSingleExpertInt8);
        EXPECT_EQ(family(int8), fam::quant_family);
        EXPECT_EQ(family(w4a8), fam::none);
        EXPECT_EQ(family(bf16), fam::none);
        EXPECT_EQ(pick_reason(w4a8), "auto_rule07_prompt_seq");
    }
    {
        Knob k(0);
        EXPECT_EQ(family(int8), fam::none);
        EXPECT_EQ(family(bf16), fam::none);
    }
    {
        Knob k(mm::kSingleExpertAll);
        EXPECT_EQ(family(int8), fam::quant_family);
        EXPECT_EQ(family(w4a8), fam::quant_family);
        EXPECT_EQ(family(bf16), fam::float_family);
    }
    if (std::getenv("ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3") == nullptr) {
        moe_test_utils::AlgoLegacyOverride legacy(true);
        EXPECT_EQ(family(bf16), fam::float_family);
        EXPECT_EQ(family(int8), fam::none);
        EXPECT_EQ(family(w4a8), fam::none);
    }
}

// The prompt of an INT8 / W4A8 layer runs on the prompt tile executor; a
// family it does not serve keeps the shape rules in both phases.  BF16 is
// unaffected.
TEST_F(MoeAlgo3Quant, PromptTileGatesTheQuantFamilies) {
    namespace pt = mm::prompt_tile;
    using fam = mm::grp_matmul_algo3_family;
    const auto family = [](const AutoProbeShape &s) {
        return mm::grp_matmul_auto_algo3_family(s.M, s.N, s.K, s.params);
    };
    const AutoProbeShape bf16 = build_auto_probe(512, 2048, 1536, 8, 32);
    const AutoProbeShape int8 = quant_layer(quant_t::dq_int8, 512, 8, 8);
    const AutoProbeShape w4a8 = quant_layer(quant_t::w4a8, 512, 8, 8);
    {
        pt::test_api::ScopedOverride mask(pt::kEnable, 9);
        EXPECT_EQ(family(int8), fam::none);
        EXPECT_EQ(family(w4a8), fam::none);
        EXPECT_EQ(family(bf16), fam::float_family);
        EXPECT_EQ(pick_reason(int8), "auto_rule07_prompt_seq");
    }
    {
        pt::test_api::ScopedOverride mask(pt::kEnable, 2);
        EXPECT_EQ(family(int8), fam::quant_family);
        EXPECT_EQ(family(w4a8), fam::none);
    }
    {
        moe_test_utils::NTileStrategyOverride rounds(1);
        EXPECT_EQ(family(int8), fam::none);
        EXPECT_EQ(family(bf16), fam::float_family);
    }
}

// The WEIGHT_CACHE=2 global-pin + phase-split guard rejects only weights one
// phase could rewrite in place: BF16 / FP16 custom-kernel packs and S8
// weights with a declared writable capacity, whatever the AUTO-only MoE knobs
// say.  INT8 / W4A8 custom-kernel packs are always out of place.
TEST_F(MoeAlgo3Quant, Wc2PhaseSplitGuardOnlyRejectsMutableWeights) {
    WeightCacheGuard wc(2);
    moe_test_utils::AlgoEnvGuard global_pin(3);
    moe_test_utils::AutoPromptAlgoOverride prompt_pin(1);
    ASSERT_TRUE(mm::grp_matmul_wc2_global_phase_split_unsupported());
    const auto hazard = [](AutoProbeShape s, size_t capacity) {
        for (auto &p : s.params) {
            p.weight_cache_type = 2;
            p.wei_buffer_capacity_bytes = capacity;
        }
        return mm::grp_matmul_wc2_global_phase_split_unsupported_for_call(
                s.M, s.N, s.K, s.params, s.num_threads);
    };
    for (quant_t q : {quant_t::dq_int8, quant_t::s8_src, quant_t::w4a8}) {
        SCOPED_TRACE(name_of(q));
        EXPECT_FALSE(hazard(quant_layer(q, 512, 8, 8), 0));
    }
    EXPECT_TRUE(hazard(quant_layer(quant_t::s8_src, 512, 8, 8), 1 << 22));
    EXPECT_TRUE(hazard(build_auto_probe(512, 2048, 1536, 8, 32), 0));
    // The AUTO-only dtype knob does not change who may write the buffer.
    const int prev
            = mm::test_api::s_grp_matmul_multi_expert_algo3_override.exchange(
                    mm::kSingleExpertW4a8, std::memory_order_relaxed);
    EXPECT_TRUE(hazard(build_auto_probe(512, 2048, 1536, 8, 32), 0));
    EXPECT_TRUE(hazard(quant_layer(quant_t::s8_src, 512, 8, 8), 1 << 22));
    EXPECT_FALSE(hazard(quant_layer(quant_t::s8_src, 512, 8, 8), 0));
    mm::test_api::s_grp_matmul_multi_expert_algo3_override.store(
            prev, std::memory_order_relaxed);
}
TEST(MoeAlgo3QuantDefaults, PromptTileCoversEveryFamily) {
    namespace pt = mm::prompt_tile;
    EXPECT_EQ(pt::defaults()[pt::kEnable], 15);
    for (pt::kernel_type_t f :
            {pt::kernel_type_t::kBf16, pt::kernel_type_t::kInt8,
                    pt::kernel_type_t::kW4a8, pt::kernel_type_t::kF16})
        EXPECT_TRUE((pt::defaults()[pt::kEnable] >> static_cast<int>(f)) & 1);
}

} // namespace
