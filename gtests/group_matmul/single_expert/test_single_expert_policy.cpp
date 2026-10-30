/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_single_expert_policy.cpp
/// @brief Single-expert grouped-matmul policy tests: the dense
///        single-expert N split, Rule 0.45 (single dense expert decode ->
///        ALGO 3), Rule 0.46 (single-expert custom-kernel phase knobs,
///        `..._FORCE_CK` and the dtype masks) and the selector shape gates.
///        Policy code lives in
///        `lowoha_operators/matmul/group_matmul/single_expert/`.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <type_traits>

#include <omp.h>

#include "auto_probe_test_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "gtest_utils.hpp"
#include "moe_test_utils.hpp"

#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/f16_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/group_matmul_m_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/group_matmul_n_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_hooks.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

using namespace auto_probe_test_utils;

// Only an act=none BF16 prompt (even K, no bias) or FP16 prompt walks K in
// chunks; the default chunk engages once the B of one call overflows L2.
TEST(TestSingleExpertPromptKc, EligibilityAndDefaultChunk) {
    using namespace zendnnl::lowoha::matmul;
    moe_test_utils::SingleExpertPromptKcOverride kc(4096);
    EXPECT_TRUE(se_prompt_kc_eligible(true, false, 16384, false, false));
    EXPECT_FALSE(se_prompt_kc_eligible(true, false, 16384, true, false));
    EXPECT_FALSE(se_prompt_kc_eligible(true, false, 16384, false, true));
    EXPECT_FALSE(se_prompt_kc_eligible(true, false, 16385, false, false));
    EXPECT_TRUE(se_prompt_kc_eligible(false, true, 16385, false, true));
    EXPECT_FALSE(se_prompt_kc_eligible(false, true, 16384, true, false));
    EXPECT_FALSE(se_prompt_kc_eligible(false, false, 16384, false, false));

    const int64_t budget = 1000000;
    // BF16: one 32-column block, 64 bytes per K row.
    EXPECT_EQ(se_prompt_kc(true, false, 32, 16384, budget), 4096);
    EXPECT_EQ(se_prompt_kc(true, false, 32, 14336, budget), 0);
    // FP16: two blocks, 128 bytes per K row.
    EXPECT_EQ(se_prompt_kc(true, true, 32, 8192, budget), 4096);
    EXPECT_EQ(se_prompt_kc(true, true, 32, 7680, budget), 0);
    EXPECT_EQ(se_prompt_kc(false, false, 32, 32768, budget), 0);
    EXPECT_EQ(se_prompt_kc(true, false, 32, 4096, 0), 0);
}

// The prompt tile's `kc` knob replaces the default chunk of a tile that can
// chunk; a depth of K or more keeps the full K.
TEST(TestSingleExpertPromptKc, TileKnobOverridesTheDefault) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    using zendnnl::lowoha::matmul::se_prompt_tile_kc;
    ck::CallContext ctx;
    ctx.prompt_kc = 4096;
    ctx.prompt_kc_eligible = true;
    EXPECT_EQ(se_prompt_tile_kc(ctx, 0, 16384), 4096);
    EXPECT_EQ(se_prompt_tile_kc(ctx, 1024, 16384), 1024);
    EXPECT_EQ(se_prompt_tile_kc(ctx, 16384, 16384), 0);
    ctx.prompt_kc = 0;
    ctx.prompt_kc_eligible = false;
    EXPECT_EQ(se_prompt_tile_kc(ctx, 1024, 16384), 0);
}

// A prompt the prompt tile executor does not take drops the prompt MR
// bound, K chunk and FP16 prompt kernel.
TEST(TestSingleExpertPromptKc, LeavingThePromptTileRestoresTheCall) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    ck::CallContext ctx;
    ctx.base_max_mr = 8;
    ctx.max_mr = ck::kMaxF16PromptMr;
    ctx.prompt_kc = 4096;
    ctx.prompt_kc_eligible = true;
    ctx.f16_prompt = true;
    zendnnl::lowoha::matmul::se_leave_prompt_tile(ctx);
    EXPECT_EQ(ctx.max_mr, 8);
    EXPECT_EQ(ctx.prompt_kc, 0);
    EXPECT_FALSE(ctx.prompt_kc_eligible);
    EXPECT_FALSE(ctx.f16_prompt);

    // A decode policy's bound (the prompt kernels did not raise it) stays.
    ctx.base_max_mr = 0;
    ctx.max_mr = 12;
    zendnnl::lowoha::matmul::se_leave_prompt_tile(ctx);
    EXPECT_EQ(ctx.max_mr, 12);
}

// The balanced remainder split is a dense single-expert optimization. Its
// default must remain the legacy split so multi-expert callers are unchanged.
TEST(TestAlignedNSplitSingleExpertScope, DefaultKeepsLegacyTailRemainder) {
    using zendnnl::lowoha::matmul::aligned_n_split;
    EXPECT_EQ(aligned_n_split(3584, 24, 0, 32), std::make_pair(0, 128));
    EXPECT_EQ(aligned_n_split(3584, 24, 23, 32), std::make_pair(2944, 3584));
}

TEST(TestAlignedNSplitSingleExpertScope, OptInBalancesAlignedBlocks) {
    using zendnnl::lowoha::matmul::aligned_n_split;
    EXPECT_EQ(aligned_n_split(3584, 24, 0, 32, true), std::make_pair(0, 160));
    EXPECT_EQ(aligned_n_split(3584, 24, 15, 32, true),
            std::make_pair(2400, 2560));
    EXPECT_EQ(aligned_n_split(3584, 24, 16, 32, true),
            std::make_pair(2560, 2688));
    EXPECT_EQ(aligned_n_split(3584, 24, 23, 32, true),
            std::make_pair(3456, 3584));
}

TEST(TestAlignedNSplitSingleExpertScope, SurplusThreadsReceiveEmptyRanges) {
    using zendnnl::lowoha::matmul::aligned_n_split;
    EXPECT_EQ(aligned_n_split(2048, 64, 0, 64, true), std::make_pair(0, 64));
    EXPECT_EQ(aligned_n_split(2048, 64, 31, 64, true),
            std::make_pair(1984, 2048));
    EXPECT_EQ(aligned_n_split(2048, 64, 32, 64, true),
            std::make_pair(2048, 2048));
    EXPECT_EQ(aligned_n_split(2048, 64, 63, 64, true),
            std::make_pair(2048, 2048));
}

// A single DQ-INT8 expert (a dense layer) runs the prompt tile under
// FORCE_CK (the layer is narrower than decode's N-tile count), whatever the
// family mask, and writes exactly what the round executor writes.
TEST(TestGroupMatmulInt8PromptTile, SingleExpertMatchesRounds) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    const int dim = 256, K = 512;
    SingleExpertCkPolicy force(1, 1, 1);
    for (const int act_int : {0, 1}) {
        for (const int rows : {300, 2048}) {
            SCOPED_TRACE(
                    ::testing::Message() << "act=" << act_int << " M=" << rows);
            const int N_op1 = act_int == 0 ? dim : 2 * dim;
            tensor_factory_t tf;
            const Int8AlgoCustomParam p {act_int, rows, 1, dim};
            std::vector<tensor_t> src, wei, bias, dst;
            ASSERT_NO_FATAL_FAILURE(build_int8_dq_inputs(
                    tf, p, N_op1, K, src, wei, bias, dst, data_type_t::f32));
            bias = {tensor_t {}};
            dst = {tf.zero_tensor(
                    {static_cast<uint64_t>(rows), static_cast<uint64_t>(N_op1)},
                    data_type_t::bf16)};
            CustomKernelInt8Override ck_int8(true);
            grp_matmul_gated_act_params act {};
            act.act = static_cast<grp_matmul_gated_act_t>(act_int);
            expect_prompt_tile_matches_rounds(/*mask=*/9, /*mb=*/0,
                    /*ns=*/0, src, wei, bias, dst,
                    act_int == 0 ? nullptr : &act, {rows}, data_type_t::s8);
        }
    }
}

// ── Rule 0.45 — single dense expert decode → ALGO 3 ──────────────────
// A lone expert (`num_ops == 1`) in decode is routed to ALGO 3 so it
// reaches the N-tile planner where the dense-FFN optimisations live
// (adaptive tiling, ragged-N fallback, auto K-blocking).  Without the
// rule, Rule 0.5's few-expert occupancy test sends one active expert to
// ALGO 1, which never touches that planner. The three tests below and
// `SingleDenseExpertDecodeEnvOffFallsBackToAlgo1` pin the rule's default
// and each of its three gates; all use the same single-expert decode probe
// so only the gate under test varies.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertDecodeRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No phase override and no DenseDecodeNTileOverride — this is the
    // out-of-the-box default (env knob unset → ON).

    // Dense-FFN W2-style decode: one expert, deep K, shallow M.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(group_matmul_algo_select(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "Rule 0.45 (default ON) must route a single-expert decode "
               "frame to ALGO 3 instead of Rule 0.5's low-occupancy ALGO 1";
}

// An explicit decode pin outranks the default, so `AUTO_DECODE_ALGO=2`
// must still win.  Pinned to 2 rather than 1 so a pass cannot be confused
// with the ALGO 1 safety fallback.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertDecodeHonoursExplicitPin) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin_m_tile(2);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(group_matmul_algo_select(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            2)
            << "an explicit AUTO_DECODE_ALGO pin must outrank Rule 0.45";
}

// Safety clamp: AUTO honours `n_tile_safe`.  A non-row-major layout is
// n_tile-unsafe, so the frame must not be handed to ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv,
        SingleDenseExpertDecodeHonoursNTileSafetyClamp) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    for (auto &c : s.layout)
        c = 'c';
    EXPECT_EQ(group_matmul_algo_select(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "Rule 0.45 must honour the n_tile_safe clamp rather than hand "
               "an unsafe shape to ALGO 3";
}

// Env gate off (`ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE=0`) → the rule is
// skipped and the frame falls through to Rule 0.5's low-occupancy ALGO 1
// arm. This is the A/B escape hatch the knob exists for.
TEST(TestGroupMatmulAutoPhaseEnv,
        SingleDenseExpertDecodeEnvOffFallsBackToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    DenseDecodeNTileOverride rule_off(0);
    SingleExpertCkPolicy single_expert_off(0, 0, 0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(group_matmul_algo_select(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "DENSE_DECODE_NTILE=0 must restore the generic few-expert "
               "occupancy decision (Rule 0.5 → ALGO 1)";
}

// Phase gate: the rule is decode-only, so a single-expert PROMPT frame
// is NOT pulled into ALGO 3 — it keeps the prompt policy and lands on
// Rule 0.7's `kShallowM → ALGO 1`.  One expert needs only `ceil(M /
// slice_target)` M-slices, far fewer than the team, so `kShallowM` (whose
// ALGO 1 gives the full team an N-split) is the right prompt answer.
// That same classification in DECODE is what Rule 0.45 exists to
// override, so this test also pins the boundary between the two.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertPromptStaysOnWideNAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    SingleExpertCkPolicy single_expert_off(0, 0, 0);

    // Same expert, prompt-class M (> kDecodeMaxM).
    auto s = build_auto_probe(/*M=*/256, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(group_matmul_algo_select(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "Rule 0.45 is decode-only; a single-expert prompt frame must "
               "stay on the prompt policy (kShallowM → ALGO 1), not ALGO 3";
}

// ── Rule 0.46 — single dense expert → ALGO 3 (custom kernel) ───────────
// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_{PROMPT,DECODE}_TILE` put the prompt /
// decode of a one-expert layer on the custom kernel, `..._FORCE_CK` both; all
// three default to 1.
static int select_one(const AutoProbeShape &s,
        zendnnl::lowoha::matmul::grp_matmul_gated_act_t act
        = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::none,
        bool epilogue_ok = true) {
    return zendnnl::lowoha::matmul::group_matmul_algo_select(s.layout, s.M, s.N,
            s.K, s.params, s.num_threads, nullptr, act, epilogue_ok);
}

namespace {

// Rewrite a BF16 auto-select probe as a DQ-INT8 call (bf16 src quantized per
// token at run time, s8 weight with a per-channel scale).
void apply_dq_int8_auto_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    static std::vector<float> wei_scale(1 << 16, 1.0f / 127.0f);
    for (size_t i = 0; i < s.params.size(); ++i) {
        auto &p = s.params[i];
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.quant_params.src_scale.dims = {static_cast<int64_t>(s.M[i]), 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {1, static_cast<int64_t>(s.N[i])};
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = wei_scale.data();
    }
}

// Rewrite a BF16 auto-select probe as a static U8xS8 call (per-tensor src
// scale, u8 src, s8 weight).
void apply_static_u8s8_auto_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    static float scale = 1.0f / 127.0f;
    static int32_t zp = 0;
    for (auto &p : s.params) {
        p.dtypes.src = data_type_t::u8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.compute = data_type_t::none;
        p.dynamic_quant = false;
        p.quant_params.src_scale.dims = {1};
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.buff = &scale;
        p.quant_params.src_zp.dims = {1};
        p.quant_params.src_zp.dt = data_type_t::s32;
        p.quant_params.src_zp.buff = &zp;
        p.quant_params.wei_scale.dims = {1};
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = &scale;
    }
}

const char *select_reason(const AutoProbeShape &s) {
    zendnnl::lowoha::matmul::GroupMatmulAlgoTrace trace;
    zendnnl::lowoha::matmul::group_matmul_algo_select(
            s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace);
    return trace.reason != nullptr ? trace.reason : "";
}

} // namespace

TEST(TestGroupMatmulAutoPhaseEnv, SingleExpertCkKnobDefaults) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    SingleExpertCkPolicy env(-1, -1, -1);
    // Every prompt and decode on the custom kernel at any N.
    if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE")
            == nullptr) {
        EXPECT_EQ(get_grp_matmul_single_expert_prompt_tile_dtypes(),
                kSingleExpertAll);
    }
    if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE")
            == nullptr) {
        EXPECT_EQ(get_grp_matmul_single_expert_decode_tile_dtypes(),
                kSingleExpertAll);
    }
    if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK") == nullptr) {
        EXPECT_EQ(get_grp_matmul_single_expert_force_ck_dtypes(),
                kSingleExpertAll);
    }
}

TEST(TestGroupMatmulAutoPhaseEnv, SingleExpertCkDtypeGrammar) {
    using namespace zendnnl::lowoha::matmul;
    static constexpr unsigned kDefault = kSingleExpertW4a8 | kSingleExpertInt8;
    const auto parse = [](const char *v) {
        return parse_grp_matmul_single_expert_dtypes(v, kDefault);
    };
    const auto unknown_of = [](const char *v) {
        std::string unknown;
        parse_grp_matmul_single_expert_dtypes(v, kDefault, &unknown);
        return unknown;
    };
    EXPECT_EQ(parse(nullptr), kDefault) << "unset";
    for (const char *all : {"1", "all", "ALL", "on", "true", "Yes"})
        EXPECT_EQ(parse(all), kSingleExpertAll) << all;
    for (const char *none : {"0", "none", "off", "FALSE", "no", " 0\n", "0\r"})
        EXPECT_EQ(parse(none), 0u) << none;
    EXPECT_EQ(parse("bf16"), kSingleExpertBf16);
    EXPECT_EQ(parse("f16"), kSingleExpertF16);
    EXPECT_EQ(parse("int8"), kSingleExpertInt8);
    EXPECT_EQ(parse("u8s8"), kSingleExpertU8s8);
    EXPECT_EQ(parse("w4a8"), kSingleExpertW4a8);
    EXPECT_EQ(parse("bf16,w4a8"), kSingleExpertBf16 | kSingleExpertW4a8);
    EXPECT_EQ(parse(" Int8 + U8S8 "), kSingleExpertInt8 | kSingleExpertU8s8);
    EXPECT_EQ(parse("bf16 w4a8\n"), kSingleExpertBf16 | kSingleExpertW4a8);
    EXPECT_EQ(parse("none,bf16"), kSingleExpertBf16) << "tokens are OR-ed";
    EXPECT_EQ(parse("w4a8,,fp16"), kSingleExpertW4a8);
    EXPECT_EQ(unknown_of("w4a8,,fp16"), "fp16");
    EXPECT_EQ(parse("fp16 s8"), kDefault) << "no known token keeps the default";
    EXPECT_EQ(unknown_of("fp16 s8"), "fp16 s8");
    EXPECT_EQ(parse("2"), kDefault) << "not a bitmask";
    EXPECT_EQ(parse("00"), kDefault);
    EXPECT_EQ(parse(""), kDefault);
    EXPECT_EQ(unknown_of(""), "<empty>");
    EXPECT_EQ(unknown_of(" , "), "<empty>");
    EXPECT_EQ(unknown_of("bf16"), "");
    EXPECT_EQ(unknown_of(nullptr), "");
}

TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertBf16PromptRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::bf16_isa_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);

    // 72 NR32 tiles: at least two per thread for 8 threads, not for 64.
    auto s = build_auto_probe(/*M=*/2048, /*K=*/768, /*N=*/2304,
            /*num_ops=*/1, /*num_threads=*/8);
    SingleExpertCkPolicy prompt_only(1, 0, 0);
    {
        auto narrow = s;
        narrow.num_threads = 64;
        EXPECT_EQ(select_one(narrow), 1)
                << "below decode's N-tile count the phase knobs do not apply";
        SingleExpertCkPolicy force(1, 1, 1);
        EXPECT_EQ(select_one(narrow), 3) << "FORCE_CK covers any N";
    }
    EXPECT_EQ(select_one(s), 3) << "plain GEMM (a routed linear)";
    EXPECT_EQ(select_one(s, grp_matmul_gated_act_t::gelu_and_mul), 3)
            << "gated FFN W13 pass";
    {
        SingleExpertCkPolicy off(0, 1, 0);
        EXPECT_EQ(select_one(s, grp_matmul_gated_act_t::gelu_and_mul), 1)
                << "prompt knob off keeps Rule 0.7's ALGO 1";
    }
    {
        AutoPromptAlgoOverride pin(2);
        EXPECT_EQ(select_one(s), 2)
                << "an explicit AUTO_PROMPT_ALGO pin must outrank Rule 0.46";
    }
    {
        CustomKernelOverride ck_off(false);
        EXPECT_EQ(select_one(s), 1)
                << "without the custom kernel there is no prompt tile path";
    }
}

// Under FORCE_CK Rule 0.46 has no size gate.  It still needs one expert, an
// epilogue the CK takes, an N the CK pack divides, a dtype the CK serves and a
// row-major layout.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertPromptScope) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::bf16_isa_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    SingleExpertCkPolicy force(1, 1, 1);
    const auto probe = [](int K, int N, int num_ops = 1) {
        return build_auto_probe(
                /*M=*/2048, K, N, num_ops, /*num_threads=*/32);
    };

    EXPECT_EQ(select_one(probe(768, 1280)), 3) << "embeddinggemma QKV";
    EXPECT_EQ(select_one(probe(1152, 768)), 3) << "embeddinggemma W2";
    EXPECT_EQ(select_one(
                      probe(4096, 28672), grp_matmul_gated_act_t::silu_and_mul),
            3)
            << "a Llama-8B W13 has no size gate";
    EXPECT_EQ(select_one(probe(768, 2304), grp_matmul_gated_act_t::gelu_and_mul,
                      /*epilogue_ok=*/false),
            1)
            << "an epilogue the CK refuses keeps AOCL";
    EXPECT_EQ(select_one(probe(768, 2304 + 16)), 1)
            << "N not divisible by the CK pack";

    auto f32 = probe(768, 2304);
    apply_shape_rules_probe(f32);
    EXPECT_EQ(select_one(f32), 1) << "the custom kernel serves no F32";

    auto unsafe = probe(768, 2304);
    unsafe.layout[0] = 'c';
    EXPECT_EQ(select_one(unsafe), 1) << "n_tile_safe clamp";

    auto two = probe(768, 2304, /*num_ops=*/2);
    int two_off = 0;
    {
        SingleExpertCkPolicy off(0, 0, 0);
        two_off = select_one(two);
    }
    EXPECT_EQ(select_one(two), two_off) << "two experts are outside Rule 0.46";
}

namespace {

enum class se_fam_t { bf16, dq_int8, w4a8, static_u8s8, f16 };

const char *se_fam_name(se_fam_t f) {
    switch (f) {
        case se_fam_t::bf16: return "BF16";
        case se_fam_t::dq_int8: return "DQ-INT8";
        case se_fam_t::w4a8: return "W4A8";
        case se_fam_t::static_u8s8: return "static U8xS8";
        default: return "FP16";
    }
}

void apply_se_fam(AutoProbeShape &s, se_fam_t f) {
    using namespace zendnnl::lowoha::matmul;
    switch (f) {
        case se_fam_t::bf16: break;
        case se_fam_t::dq_int8: apply_dq_int8_auto_probe(s); break;
        case se_fam_t::w4a8: apply_w4a8_auto_probe(s); break;
        case se_fam_t::static_u8s8: apply_static_u8s8_auto_probe(s); break;
        case se_fam_t::f16:
            for (auto &p : s.params) {
                p.dtypes.src = data_type_t::f16;
                p.dtypes.wei = data_type_t::f16;
                p.dtypes.dst = data_type_t::f16;
                p.dtypes.bias = data_type_t::f16;
            }
            break;
    }
}

#define SKIP_IF_NO_SERVED_SE_FAMS() \
    do { \
        if (served_se_fams().empty()) \
            GTEST_SKIP() << "the custom kernel serves no dtype on this host"; \
    } while (0)

std::vector<se_fam_t> served_se_fams();

std::vector<se_fam_t> served_se_fams() {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    std::vector<se_fam_t> fams;
    if (ck::bf16_isa_supported()) fams.push_back(se_fam_t::bf16);
    if (ck::avx512vnni_available()) {
        fams.push_back(se_fam_t::dq_int8);
        fams.push_back(se_fam_t::w4a8);
        fams.push_back(se_fam_t::static_u8s8);
    }
    if (ck::avx512f16_available()) fams.push_back(se_fam_t::f16);
    return fams;
}

} // namespace

// Unset `ZENDNNL_GRP_MATMUL_ALGO` runs a dense single expert on the custom
// kernel in both phases; an explicit `0` keeps its prompt on AOCL-DLP
// (Rule 0.7, ALGO 1) while the decode stays on the custom kernel.
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitAlgoZeroKeepsTheSingleExpertPrompt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE") != nullptr
            || std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE")
                    != nullptr) {
        GTEST_SKIP() << "single-expert knobs set in the environment";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override int8_on(true);
    CustomKernelF16Override f16_on(true);
    SingleExpertCkPolicy env(-1, -1, -1);
    {
        AlgoLegacyOverride legacy(true);
        EXPECT_EQ(get_grp_matmul_single_expert_prompt_tile_dtypes(), 0u);
        EXPECT_EQ(get_grp_matmul_single_expert_decode_tile_dtypes(),
                static_cast<unsigned>(kSingleExpertAll));
    }
    EXPECT_FALSE(get_grp_matmul_algo_legacy()) << "an AlgoEnvGuard pin";
    SKIP_IF_NO_SERVED_SE_FAMS();
    for (const se_fam_t f : served_se_fams()) {
        if (f == se_fam_t::static_u8s8) continue;
        SCOPED_TRACE(se_fam_name(f));
        auto prompt = build_auto_probe(/*M=*/1024, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/8);
        auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/8);
        apply_se_fam(prompt, f);
        apply_se_fam(decode, f);
        {
            AlgoLegacyOverride legacy(false);
            EXPECT_EQ(select_one(prompt), 3);
            EXPECT_STREQ(
                    select_reason(prompt), "auto_rule046_single_dense_prompt");
        }
        AlgoLegacyOverride legacy(true);
        EXPECT_EQ(select_one(prompt), 1);
        EXPECT_STREQ(select_reason(prompt), "auto_rule07_prompt_seq");
        EXPECT_EQ(select_one(decode), 3);
        EXPECT_STREQ(select_reason(decode), "auto_rule046_single_dense_decode");
    }
}

// Each phase knob covers its phase for every dtype the custom kernel serves,
// independently of the other, at an N with decode's N-tile count; with both
// on, the layer stays on ALGO 3 even with Rule 0.45 off and at one thread.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertPerPhaseKnobs) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override int8_on(true);
    CustomKernelF16Override f16_on(true);
    DenseDecodeNTileOverride rule045_off(0);
    const char *kPrompt = "auto_rule046_single_dense_prompt";
    const char *kDecode = "auto_rule046_single_dense_decode";

    SKIP_IF_NO_SERVED_SE_FAMS();
    for (const se_fam_t f : served_se_fams()) {
        SCOPED_TRACE(se_fam_name(f));
        // N=1280 is 40 NR32 tiles: wide enough for 8 threads, not for 32.
        auto prompt = build_auto_probe(/*M=*/1024, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/8);
        auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/8);
        auto one_thread = build_auto_probe(/*M=*/4, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/1);
        auto one_thread_prompt = build_auto_probe(/*M=*/1024, /*K=*/1280,
                /*N=*/1280, /*num_ops=*/1, /*num_threads=*/1);
        for (auto *s : {&prompt, &decode, &one_thread, &one_thread_prompt})
            apply_se_fam(*s, f);
        // The prompt tile does not take static U8xS8; FORCE_CK must list it.
        const bool prompt_tileable = f != se_fam_t::static_u8s8;
        for (const auto &[pk, dk] : {std::pair {1, 0}, std::pair {0, 1},
                     std::pair {1, 1}, std::pair {0, 0}}) {
            SCOPED_TRACE(::testing::Message()
                    << "prompt=" << pk << " decode=" << dk);
            SingleExpertCkPolicy policy(pk, dk, 0);
            if (pk && prompt_tileable) {
                EXPECT_EQ(select_one(prompt), 3);
                EXPECT_STREQ(select_reason(prompt), kPrompt);
            } else {
                EXPECT_EQ(select_one(prompt), 1) << "Rule 0.7";
                EXPECT_STREQ(select_reason(prompt), "auto_rule07_prompt_seq");
            }
            if (dk) {
                EXPECT_EQ(select_one(decode), 3);
                EXPECT_STREQ(select_reason(decode), kDecode);
                EXPECT_EQ(select_one(one_thread), 3) << "at any team size";
            } else {
                EXPECT_EQ(select_one(decode), 1) << "Rule 0.45 off";
                EXPECT_EQ(select_one(one_thread), 1) << "one thread";
            }
            // A one-thread call keeps ALGO 3 when the custom kernel covers
            // its own phase: a covered prompt never falls back to AOCL.
            EXPECT_EQ(select_one(one_thread_prompt),
                    pk && prompt_tileable ? 3 : 1);
            // At 32 threads the same N is below decode's N-tile count.
            auto narrow_prompt = prompt, narrow_decode = decode;
            narrow_prompt.num_threads = narrow_decode.num_threads = 32;
            EXPECT_EQ(select_one(narrow_prompt), 1);
            EXPECT_EQ(select_one(narrow_decode), 1);
        }
    }

    // Prompt exclusions keep the prompt knob off the prompt, FORCE_CK or not:
    // it lifts only the N gate.
    if (!custom_kernel::bf16_isa_supported()) return;
    auto prompt = build_auto_probe(/*M=*/1024, /*K=*/768, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/8);
    {
        NTileStrategyOverride decode_d(1);
        {
            SingleExpertCkPolicy prompt_only(1, 0, 0);
            EXPECT_EQ(select_one(prompt), 1)
                    << "decode_d never serves the tile";
        }
        SingleExpertCkPolicy force(1, 1, 1);
        EXPECT_EQ(select_one(prompt), 1);
    }
    {
        NTileStrategyOverride strategy0(0);
        SingleExpertCkPolicy force(1, 1, 1);
        EXPECT_EQ(select_one(prompt), 1)
                << "strategy 0 plans a lone expert as Sequential AOCL";
    }
    {
        PromptTileOverride tile_off(0);
        SingleExpertCkPolicy forced(1, 1, 1);
        EXPECT_EQ(select_one(prompt), 1)
                << "PROMPT_TILE=0 keeps every prompt off the tile";
    }

    // AOCL-prepacked: the custom kernel converts BF16 (both phases), not a
    // quantized symmetric pack.
    SingleExpertCkPolicy both(1, 1, 0);
    auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/8);
    auto prepacked = decode;
    prepacked.params[0].mem_format_b = 'r';
    EXPECT_STREQ(select_reason(prepacked), kDecode);
    if (custom_kernel::avx512vnni_available()) {
        auto int8_prepacked = prepacked;
        apply_dq_int8_auto_probe(int8_prepacked);
        EXPECT_EQ(select_one(int8_prepacked), 1);
        auto int8 = decode;
        apply_dq_int8_auto_probe(int8);
        CustomKernelInt8Override int8_off(false);
        EXPECT_EQ(select_one(int8), 1)
                << "the INT8 sub-knob gates the INT8 layer";
    }

    auto pooled = build_auto_probe(/*M=*/1024, /*K=*/768, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    pooled.params[0].active_matmul = 1;
    pooled.params[0].total_matmul = 8;
    EXPECT_STREQ(select_reason(pooled), "auto_algo3")
            << "one active expert of an MoE pool follows its MoE layer";
}

// Each knob covers exactly the dtypes it lists: `FORCE_CK=w4a8` or
// `FORCE_CK=bf16,int8` with the phase knobs off puts only those dtypes on the
// custom kernel; a phase knob listing another dtype covers that phase for it.
TEST(TestGroupMatmulAutoPhaseEnv, SingleExpertCkDtypeSelection) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override int8_on(true);
    CustomKernelF16Override f16_on(true);
    DenseDecodeNTileOverride rule045_off(0);
    const auto bit = [](se_fam_t f) {
        switch (f) {
            case se_fam_t::bf16: return kSingleExpertBf16;
            case se_fam_t::dq_int8: return kSingleExpertInt8;
            case se_fam_t::w4a8: return kSingleExpertW4a8;
            case se_fam_t::static_u8s8: return kSingleExpertU8s8;
            default: return kSingleExpertF16;
        }
    };
    const char *kForce = "auto_rule046_single_expert_force_ck";
    const char *kForceDecode = "auto_rule046_single_expert_force_ck_decode";
    SKIP_IF_NO_SERVED_SE_FAMS();

    for (const unsigned force : {unsigned(kSingleExpertW4a8),
                 unsigned(kSingleExpertBf16 | kSingleExpertInt8),
                 unsigned(kSingleExpertU8s8 | kSingleExpertF16)}) {
        SCOPED_TRACE(::testing::Message() << "force mask " << force);
        SingleExpertCkMasks masks(kSingleExpertAll, kSingleExpertAll, force);
        for (const se_fam_t f : served_se_fams()) {
            SCOPED_TRACE(se_fam_name(f));
            auto prompt = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, /*num_threads=*/32);
            auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, /*num_threads=*/32);
            apply_se_fam(prompt, f);
            apply_se_fam(decode, f);
            if ((force & bit(f)) != 0) {
                // The prompt tile never takes static U8xS8.
                if (f == se_fam_t::static_u8s8)
                    EXPECT_STREQ(
                            select_reason(prompt), "auto_rule07_prompt_seq");
                else
                    EXPECT_STREQ(select_reason(prompt), kForce);
                EXPECT_STREQ(select_reason(decode), kForceDecode);
            } else {
                EXPECT_EQ(select_one(prompt), 1);
                EXPECT_STREQ(select_reason(prompt), "auto_rule07_prompt_seq");
                EXPECT_EQ(select_one(decode), 1) << "Rule 0.45 off";
            }
        }
    }

    // PROMPT_TILE=bf16, DECODE_TILE=int8,w4a8, FORCE_CK=w4a8.  At 8 threads
    // N=1280 passes the N gate: the phase knobs alone decide.  At 32 it does
    // not: only the forced W4A8 decode stays on the custom kernel.
    SingleExpertCkMasks mixed(kSingleExpertBf16,
            kSingleExpertInt8 | kSingleExpertW4a8, kSingleExpertW4a8);
    for (const int threads : {8, 32}) {
        for (const se_fam_t f : served_se_fams()) {
            SCOPED_TRACE(::testing::Message()
                    << se_fam_name(f) << " threads=" << threads);
            auto prompt = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, threads);
            auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, threads);
            apply_se_fam(prompt, f);
            apply_se_fam(decode, f);
            const std::string p = select_reason(prompt);
            const std::string d = select_reason(decode);
            const bool wide = threads == 8;
            EXPECT_EQ(p == "auto_rule046_single_dense_prompt",
                    wide && f == se_fam_t::bf16);
            EXPECT_NE(p, kForce) << "FORCE_CK adds no phase";
            EXPECT_EQ(d == "auto_rule046_single_dense_decode",
                    wide && (f == se_fam_t::dq_int8 || f == se_fam_t::w4a8));
            EXPECT_EQ(d == kForceDecode, !wide && f == se_fam_t::w4a8);
        }
    }

    // Production defaults: prompt on AOCL-DLP, decode on the custom kernel
    // at any N.
    {
        SingleExpertCkMasks defaults(0u, kSingleExpertAll, kSingleExpertAll);
        for (const se_fam_t f : served_se_fams()) {
            SCOPED_TRACE(se_fam_name(f));
            auto prompt = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, /*num_threads=*/32);
            auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                    /*num_ops=*/1, /*num_threads=*/32);
            apply_se_fam(prompt, f);
            apply_se_fam(decode, f);
            EXPECT_STREQ(select_reason(prompt), "auto_rule07_prompt_seq");
            EXPECT_STREQ(select_reason(decode), kForceDecode);
        }
    }
}

// Rule 0.46 leaves the shapes the custom kernel always refuses (INT8 K not a
// VNNI quad, W4A8 groups not whole K-octets, a gated activation into a
// non-BF16 destination) on their AUTO rules, and tells DQ-INT8 from static
// U8xS8 by dynamic quantization, not by scale shape.
TEST(TestGroupMatmulAutoPhaseEnv, SingleExpertCkRefusedShapesAndDqForms) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512vnni_available()) {
        GTEST_SKIP() << "needs the INT8 / W4A8 custom kernel (AVX512-VNNI)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override int8_on(true);
    SingleExpertCkPolicy force(1, 1, 1);
    const char *kForce = "auto_rule046_single_expert_force_ck";

    auto int8_ok = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    apply_dq_int8_auto_probe(int8_ok);
    EXPECT_STREQ(select_reason(int8_ok), kForce);
    auto int8_odd_k = build_auto_probe(/*M=*/512, /*K=*/1282, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    apply_dq_int8_auto_probe(int8_odd_k);
    EXPECT_STRNE(select_reason(int8_odd_k), kForce) << "K % 4 != 0";

    auto w4_ok = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    apply_w4a8_auto_probe(w4_ok);
    w4_ok.params[0].quant_params.wei_scale.dims = {20, 1280};
    EXPECT_STREQ(select_reason(w4_ok), kForce) << "group 64";
    auto w4_bad_group = w4_ok;
    w4_bad_group.params[0].quant_params.wei_scale.dims = {320, 1280};
    EXPECT_STRNE(select_reason(w4_bad_group), kForce) << "group 4";

    if (custom_kernel::bf16_isa_supported()) {
        auto gated_f32 = build_auto_probe(/*M=*/512, /*K=*/768, /*N=*/2304,
                /*num_ops=*/1, /*num_threads=*/32);
        gated_f32.params[0].dtypes.dst = data_type_t::f32;
        EXPECT_EQ(
                select_one(gated_f32, grp_matmul_gated_act_t::gelu_and_mul), 1);
        EXPECT_EQ(select_one(gated_f32), 3) << "plain GEMM into F32 is served";
    }

    // Classification follows `is_static_u8s8_quant_config`: u8 x s8 without
    // dynamic quantization is `u8s8`, whatever the shape of its scalar
    // source scale; a dynamic BF16 source (one row included) is `int8`.
    static float one = 1.0f / 127.0f;
    auto stat = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    apply_static_u8s8_auto_probe(stat);
    auto stat_2d = stat;
    stat_2d.params[0].dtypes.compute = data_type_t::u8;
    stat_2d.params[0].quant_params.src_scale.dims = {1, 1};
    stat_2d.params[0].quant_params.src_scale.buff = &one;
    auto dq_row = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    apply_dq_int8_auto_probe(dq_row);
    {
        SingleExpertCkMasks int8_only(
                kSingleExpertInt8, kSingleExpertInt8, kSingleExpertInt8);
        EXPECT_TRUE(grp_matmul_single_expert_ck(dq_row.params[0], 1280, 1280)
                            .any());
        EXPECT_FALSE(
                grp_matmul_single_expert_ck(stat.params[0], 1280, 1280).any());
        EXPECT_FALSE(grp_matmul_single_expert_ck(stat_2d.params[0], 1280, 1280)
                             .any());
    }
    {
        SingleExpertCkMasks u8s8_only(
                kSingleExpertU8s8, kSingleExpertU8s8, kSingleExpertU8s8);
        EXPECT_FALSE(grp_matmul_single_expert_ck(dq_row.params[0], 1280, 1280)
                             .any());
        EXPECT_TRUE(
                grp_matmul_single_expert_ck(stat.params[0], 1280, 1280).any());
        EXPECT_TRUE(grp_matmul_single_expert_ck(stat_2d.params[0], 1280, 1280)
                            .any());
        auto stat_decode = stat;
        stat_decode.M[0] = 1;
        EXPECT_STREQ(select_reason(stat_decode),
                "auto_rule046_single_expert_force_ck_decode");
    }
}

// The WEIGHT_CACHE=2 guard against a phase-split global pin flags a raw
// BF16 single expert only when its both phases are on the custom kernel for
// the call's shape and thread count: the phase knobs below the N gate do not
// cover it, FORCE_CK does.
TEST(TestGroupMatmulAutoPhaseEnv, Wc2PhaseSplitGuardUsesCallShape) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::bf16_isa_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    const int32_t old_wc = cfg.get_weight_cache();
    cfg.set_weight_cache(2);
    {
        AlgoEnvGuard global_pin(1);
        AutoDecodeAlgoOverride decode_pin(3);
        CustomKernelOverride ck_on(true);
        ASSERT_TRUE(grp_matmul_wc2_global_phase_split_unsupported());
        auto s = build_auto_probe(/*M=*/4, /*K=*/768, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/32);
        s.params[0].weight_cache_type = 2;
        const auto hazard = [&](int threads) {
            return grp_matmul_wc2_global_phase_split_unsupported_for_call(
                    s.M, s.N, s.K, s.params, threads);
        };
        {
            SingleExpertCkPolicy phase_only(1, 1, 0);
            EXPECT_FALSE(hazard(32)) << "below the N gate: not covered";
            EXPECT_TRUE(hazard(8)) << "40 NR32 tiles pass the gate at 8";
        }
        {
            SingleExpertCkPolicy force(1, 1, 1);
            EXPECT_TRUE(hazard(32));
        }
        {
            SingleExpertCkPolicy off(0, 0, 0);
            EXPECT_FALSE(hazard(8));
        }
    }
    cfg.set_weight_cache(old_wc);
}

// With both phase knobs on, `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK=1`
// puts every single expert the custom kernel serves on ALGO 3 below the N
// gate, whatever its dtype; it still needs an N the CK pack divides and
// honours an explicit phase pin.
TEST(TestGroupMatmulAutoPhaseEnv, SingleExpertForceCkRoutesEveryFamily) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override int8_on(true);
    CustomKernelF16Override f16_on(true);
    DenseDecodeNTileOverride rule045_off(0);
    SingleExpertCkPolicy force(1, 1, 1);
    SKIP_IF_NO_SERVED_SE_FAMS();

    for (const se_fam_t f : served_se_fams()) {
        SCOPED_TRACE(se_fam_name(f));
        auto prompt = build_auto_probe(/*M=*/512, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/32);
        auto decode = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
                /*num_ops=*/1, /*num_threads=*/32);
        apply_se_fam(prompt, f);
        apply_se_fam(decode, f);
        // The prompt tile never takes static U8xS8.
        const bool tileable = f != se_fam_t::static_u8s8;
        EXPECT_EQ(select_one(prompt), tileable ? 3 : 1);
        EXPECT_STREQ(select_reason(prompt),
                tileable ? "auto_rule046_single_expert_force_ck"
                         : "auto_rule07_prompt_seq");
        EXPECT_EQ(select_one(decode), 3);
        EXPECT_STREQ(select_reason(decode),
                "auto_rule046_single_expert_force_ck_decode");
        {
            AutoPromptAlgoOverride pin(1);
            EXPECT_EQ(select_one(prompt), 1)
                    << "an explicit AUTO_PROMPT_ALGO pin outranks the force";
        }
        {
            SingleExpertCkPolicy off(0, 0, 0);
            EXPECT_EQ(select_one(decode), 1) << "Rule 0.45 off";
        }
        auto ragged = decode;
        ragged.N[0] = 1280 + 16;
        EXPECT_STRNE(select_reason(ragged),
                "auto_rule046_single_expert_force_ck_decode")
                << "N not divisible by the CK pack";
    }

    if (!custom_kernel::bf16_isa_supported()) return;
    auto bf16_prepacked = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    bf16_prepacked.params[0].mem_format_b = 'r';
    EXPECT_STREQ(select_reason(bf16_prepacked),
            "auto_rule046_single_expert_force_ck_decode")
            << "the custom kernel converts an AOCL-prepacked BF16 weight";
    auto two = build_auto_probe(/*M=*/1, /*K=*/1280, /*N=*/1280,
            /*num_ops=*/2, /*num_threads=*/32);
    EXPECT_STRNE(
            select_reason(two), "auto_rule046_single_expert_force_ck_decode");
}
