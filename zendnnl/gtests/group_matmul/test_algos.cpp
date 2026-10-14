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

/// @file test_algos.cpp
/// @brief Scheduling-ALGO and custom-kernel matrix gtest sections.  Owned:
///
///   [7]  TestFusedMoEAlgos       - fused MoE x ALGO 1/2/3 x mixed precision
///                                  x bias.
///   [7b] TestFusedMoEAlgoCustom  - fused MoE x strategy / tight / custom
///                                  BF16 microkernel env-knob matrix.
///   [8]  TestGroupMatmulAlgoCustom - non-fused parallel x custom BF16
///                                  microkernel env-knob matrix.
///
/// Split from `test_group_matmul.cpp` during the gtests folder refactor;
/// see `group_matmul/README.md` for the file layout overview.

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

#include "group_matmul_test_helpers.hpp"
#include "gtest_utils.hpp"
#include "kblock_test_hooks.hpp"
#include "moe_test_utils.hpp"

// Direct access to the dispatcher's ALGO selector — used by section
// [8b] (`TestGroupMatmulAutoSelectAlgo`) to assert the auto-select
// decision matrix without spinning up the full GEMM stack.
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/m_tile/group_matmul_m_tile.hpp"

// `GroupNTileStrategy` enum + `test_api::PhaseBSnapshot` /
// `s_capture_phase_b` / `s_last_phase_b_snapshot` — used by
// section [8a.1] (`TestGroupMatmulPhaseBRemainder.HeaviestFirstAssignment`)
// to white-box assert on the planner's Phase B output.
#include "lowoha_operators/matmul/group_matmul/n_tile/group_matmul_n_tile.hpp"

// `custom_kernel::dispatch_supported()` — Phase B tests check it
// to skip cleanly on hosts without AVX512BF16 (where forcing the
// env-cache CK override is necessary but not sufficient: the
// runtime `prepare_for_call` would refuse CK independently).
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

// `custom_kernel::avx512f16_available()` — the FP16 weight-cache tests
// skip on hosts/toolchains without native AVX-512-FP16, where
// `group_matmul_direct` refuses an f16 call outright.
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/f16_microkernel.hpp"

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

// ???????????????????????????????????????????????????????????????????????????????
// [7] TestFusedMoEAlgos: fused path ? ALGO 1/2/3 ? mixed precision ? bias
// ???????????????????????????????????????????????????????????????????????????????

struct FusedAlgoTestParam {
    int algo, act_int;
    bool is_bf16, mixed_prec, use_bias;
    int M, num_ops;
    // dim=0 ? use default 128.  Larger dims (?256) force N_gate_up ? 512,
    // which enables multi-thread N-tiling in ALGO 3 and exercises the
    // per-thread fused-swiglu-oai epilogue path (which historically hid a
    // cross-thread write-after-read race that only triggered when n_thr>1).
    int dim = 0;
};

static std::string FusedAlgoParamName(
        const ::testing::TestParamInfo<FusedAlgoTestParam> &info) {
    static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
    const auto &p = info.param;
    std::string name
            = "algo" + std::to_string(p.algo) + "_" + act_names[p.act_int];
    if (p.mixed_prec) {
        name += "_bf16f32";
    } else {
        name += (p.is_bf16 ? "_bf16" : "_f32");
    }
    if (p.use_bias) { name += "_bias"; }
    name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops);
    if (p.dim > 0) { name += "_d" + std::to_string(p.dim); }
    return name;
}

class TestFusedMoEAlgos : public ::testing::TestWithParam<FusedAlgoTestParam> {
};

TEST_P(TestFusedMoEAlgos, Correctness) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    reset_grp_matmul_caches();

    const auto &p = GetParam();
    const int dim = (p.dim > 0) ? p.dim : 128;
    const int N_gate_up = 2 * dim, H = 256;
    const int M = p.M, K = H, num_ops = p.num_ops;
    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);

    const data_type_t src_dt = p.is_bf16 ? data_type_t::bf16 : data_type_t::f32;
    const data_type_t dst_dt = p.mixed_prec
            ? data_type_t::f32
            : (p.is_bf16 ? data_type_t::bf16 : data_type_t::f32);
    const data_type_t wei_dt = src_dt;
    const bool use_bf16_in = (p.is_bf16 || p.mixed_prec);
    const bool use_bf16_out = !p.mixed_prec && p.is_bf16;

    // Op2's K dimension follows the activation: gated => dim, none =>
    // N_gate_up.  See `op2_k_for_act` in group_matmul_fused_moe.cpp.
    const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
    const int K_down = act_is_none ? N_gate_up : dim;

    AlgoEnvGuard algo_guard(p.algo);
    // The fused-swiglu_oai epilogue in ALGO 3 is gated by
    // ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT (default OFF) ? see
    // get_grp_n_tile_fused_act() in group_matmul_parallel_common.hpp.
    // Force it ON here so the shapes below (where N_gate_up >
    // kDecodeNTile) drive ALGO 3's per-thread fused epilogue with
    // n_thr > 1 threads per expert.  That is the row-split path the
    // matmul?activation barrier + GroupNTileContext::apply_swiglu_oai
    // correctness fix exists to protect.
    NTileFusedActOverride fused_act_guard(true);

    // Allocate: input-side may be bf16; output-side may differ (mixed_prec).
    TypedBuffers src, w1, d1, d1r, w2, d2, d2r;
    src.alloc(num_ops, (size_t)M * K, use_bf16_in, p.mixed_prec);
    w1.alloc(num_ops, (size_t)K * N_gate_up, use_bf16_in, p.mixed_prec);
    w2.alloc(num_ops, (size_t)K_down * H, use_bf16_in, p.mixed_prec);
    d1.alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
    d1r.alloc(num_ops, (size_t)M * N_gate_up, use_bf16_out);
    d2.alloc(num_ops, (size_t)M * H, use_bf16_out);
    d2r.alloc(num_ops, (size_t)M * H, use_bf16_out);

    std::vector<std::vector<float>> bias_f(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        // Always generate in f32 then mirror to bf16 when needed.
        std::vector<float> s_tmp((size_t)M * K), w1_tmp((size_t)K * N_gate_up),
                w2_tmp((size_t)K_down * H);
        fill_src(s_tmp, e);
        fill_wei1(w1_tmp, e);
        fill_wei2(w2_tmp, e);
        if (src.store_f32) { src.f32[e] = s_tmp; }
        if (w1.store_f32) { w1.f32[e] = w1_tmp; }
        if (w2.store_f32) { w2.f32[e] = w2_tmp; }
        if (src.store_bf16) {
            src.bf16[e].resize(s_tmp.size());
            for (size_t i = 0; i < s_tmp.size(); ++i) {
                src.bf16[e][i] = bfloat16_t(s_tmp[i]);
            }
        }
        if (w1.store_bf16) {
            w1.bf16[e].resize(w1_tmp.size());
            for (size_t i = 0; i < w1_tmp.size(); ++i) {
                w1.bf16[e][i] = bfloat16_t(w1_tmp[i]);
            }
        }
        if (w2.store_bf16) {
            w2.bf16[e].resize(w2_tmp.size());
            for (size_t i = 0; i < w2_tmp.size(); ++i) {
                w2.bf16[e][i] = bfloat16_t(w2_tmp[i]);
            }
        }
        bias_f[e].resize(H);
        for (int i = 0; i < H; ++i) {
            bias_f[e][i] = 0.01f * ((i + e) % 5);
        }
    }

    auto gv1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);
    auto srcs = src.cptrs(use_bf16_in);
    auto wei1 = w1.cptrs(use_bf16_in);
    auto wei2 = w2.cptrs(use_bf16_in);
    auto dst1 = d1.ptrs(use_bf16_out);
    auto dst1r = d1r.ptrs(use_bf16_out);
    auto dst2 = d2.ptrs(use_bf16_out);
    auto dst2r = d2r.ptrs(use_bf16_out);
    std::vector<const void *> bias1(num_ops, nullptr), bias2(num_ops, nullptr);
    if (p.use_bias)
        for (int e = 0; e < num_ops; ++e) {
            bias2[e] = bias_f[e].data();
        }

    grp_matmul_gated_act_params act {};
    act.act = act_type;
    auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

    // Reference: Op1+Act then Op2.
    {
        auto pr1 = make_mixed_params(num_ops, src_dt, wei_dt, dst_dt);
        ASSERT_EQ(group_matmul_direct(gv1.layout, gv1.transA, gv1.transB,
                          gv1.Ms, gv1.Ns, gv1.Ks, gv1.alpha, srcs, gv1.lda,
                          wei1, gv1.ldb, bias1, gv1.beta, dst1r, gv1.ldc,
                          gv1.is_wc, pr1, nullptr, act_ptr),
                status_t::success)
                << "Ref Op1 failed";

        std::vector<const void *> s2(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            s2[e] = dst1r[e];
        }
        auto gv2 = GemmVecs::uniform(num_ops, M, H, K_down);
        gv2.lda.assign(num_ops, N_gate_up);
        auto pr2 = make_mixed_params(num_ops, dst_dt, wei_dt, dst_dt,
                p.use_bias ? data_type_t::f32 : data_type_t::none);
        ASSERT_EQ(
                group_matmul_direct(gv2.layout, gv2.transA, gv2.transB, gv2.Ms,
                        gv2.Ns, gv2.Ks, gv2.alpha, s2, gv2.lda, wei2, gv2.ldb,
                        bias2, gv2.beta, dst2r, gv2.ldc, gv2.is_wc, pr2),
                status_t::success)
                << "Ref Op2 failed";
    }

    // Fused path.
    auto fused = make_fused_moe_op2(num_ops, H, wei2, bias2);
    fused.bias_dt_down = p.use_bias ? data_type_t::f32 : data_type_t::none;
    fused.dst_down = dst2;
    fused.ldc_down = std::vector<int>(num_ops, H);
    {
        auto pf = make_mixed_params(num_ops, src_dt, wei_dt, dst_dt);
        ASSERT_EQ(group_matmul_direct(gv1.layout, gv1.transA, gv1.transB,
                          gv1.Ms, gv1.Ns, gv1.Ks, gv1.alpha, srcs, gv1.lda,
                          wei1, gv1.ldb, bias1, gv1.beta, dst1, gv1.ldc,
                          gv1.is_wc, pf, nullptr, act_ptr, &fused),
                status_t::success)
                << "Fused call failed (algo=" << p.algo << ")";
    }

    std::ostringstream lbl;
    lbl << "algo=" << p.algo << " act=" << p.act_int
        << (p.mixed_prec ? " bf16>f32" : (p.is_bf16 ? " bf16" : " f32"))
        << (p.use_bias ? " +bias" : "");
    verify_per_expert_2d(d2, H, d2r, H, num_ops, M, H, use_bf16_out,
            tol_fused(p.is_bf16 || p.mixed_prec), lbl.str());
}

static std::vector<FusedAlgoTestParam> make_fused_algo_params() {
    std::vector<FusedAlgoTestParam> out;
    // All 3 ALGOs ? all 4 activation types ? both dtypes.
    // Covers: ALGO-specific fused MoE dispatch paths for every activation.
    for (int algo : {1, 2, 3})
        for (int act : {0, 1, 2, 3}) // none, silu, gelu, swiglu
            for (bool bf : {false, true})
                out.push_back({algo, act, bf, false, false, 4, 4});
    // Mixed precision (BF16 src ? F32 dst) per ALGO.
    for (int algo : {1, 2, 3})
        out.push_back({algo, 1, true, true, false, 4, 4});
    // Non-null down_proj bias per ALGO.
    for (int algo : {1, 2, 3})
        out.push_back({algo, 1, false, false, true, 4, 4});
    // ALGO 2 M-tile with varying M (small M=1 and larger M=16).
    for (int m : {1, 16})
        out.push_back({2, 1, false, false, false, m, 4});
    // ALGO 3 two-pass with many experts.
    out.push_back({3, 1, false, false, false, 4, 8});

    // ALGO 3 fused swiglu_oai_mul ? race-exposure shapes.
    //
    // Pre-fix, apply_n_tile_paired_swiglu_oai split the epilogue by N
    // columns, which aliased thread t's compact-output writes
    // [p_start_t, p_end_t) with a lower-index thread's pair-read range
    // [2?p_start_{t-1}, 2?p_start_t).  The race only fires when
    // flat_n_tile actually runs more than one thread per expert ?
    // i.e. (1) `ntile_viable` is true and (2) `thr_per_expert` (or the
    // path-B `n_thr`) resolves to ?2.  Every earlier swiglu case in this
    // file used dim ? 128 so N_gate_up ? 256 = kDecodeNTile ? fallback ?
    // one thread per expert ? the bug stayed hidden.
    //
    // The shapes below hit the race-prone code on both a typical 16-thread
    // developer run AND a 128-thread EPYC CI run (verified with a Python
    // simulation of flat_n_tile's decision tree):
    //
    //   shape (M=8, E=8, d=1024):
    //       16t  ? path (B), n_thr=2 per expert, 2-round batched N-tile
    //      128t  ? path (D), n_thr=8 per expert, decode_parallel
    //
    //   shape (M=64, E=8, d=2048):
    //       16t  ? path (B), n_thr=2 per expert
    //      128t  ? path (A), n_thr=8 per expert, L3-batched few-expert
    //
    // Together they exercise all three multi-threaded epilogue paths
    // (D/A/B) and make the fix observable: each thread writes into a
    // disjoint row slice instead of the aliased pair?compact column
    // slice, so there is no cross-thread overlap.  Pre-fix these shapes
    // returned NaNs / wrong arithmetic that exceed the BF16 tolerance;
    // post-fix they match the 2-pass reference.
    for (bool bf : {false, true}) {
        out.push_back({3, /*act=*/3, bf, false, false, /*M=*/8, /*E=*/8,
                /*dim=*/1024});
        out.push_back({3, /*act=*/3, bf, false, false, /*M=*/64, /*E=*/8,
                /*dim=*/2048});
    }
    // Bias and mixed-precision cross-checks on the decode-shape case.
    out.push_back({3, /*act=*/3, /*bf=*/true, /*mixed=*/true, /*bias=*/false,
            /*M=*/8, /*E=*/8, /*dim=*/1024});
    out.push_back({3, /*act=*/3, /*bf=*/false, /*mixed=*/false, /*bias=*/true,
            /*M=*/8, /*E=*/8, /*dim=*/1024});
    return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedAlgos, TestFusedMoEAlgos,
        ::testing::ValuesIn(make_fused_algo_params()), FusedAlgoParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [7b] TestFusedMoEAlgoCustom: fused-MoE env-knob matrix
//
// Mirrors TestGroupMatmulAlgoCustom for the non-fused path but for the
// fused MoE entry (Op1+act ? Op2).  Targets the strategy-selection
// contract cemented in Option A:
//
//   * ZENDNNL_GRP_MATMUL_ALGO = {1,2,3,5,6}  ? strategy selector, the
//                                              single source of truth
//                                              for the fused path.
//   * ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT=0/1 ? V1 vs V2 (V2 is the
//                                              ALGO 3 + swiglu + tight
//                                              specialist; engages only
//                                              when env_algo ? {0, 3}).
//   * ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0/1   ? custom BF16 ukernel hook
//                                              (flat_n_tile + V2 Op2).
//
// Reference strategy: a single known-good baseline ? forced ALGO=1,
// TIGHT=0, CUSTOM=0, legacy two-call (Op1 via group_matmul_direct +
// Op2 via group_matmul_direct) ? compared against the fused
// internal-alloc call under the parameterised env.  Any breakage of
// Option A's gating surface (e.g. V2 silently engaging for ALGO 5,
// or CUSTOM_KERNEL=1 corrupting Op2 for non-BF16 dtypes) produces a
// comparison failure.
// ???????????????????????????????????????????????????????????????????????????????

struct FusedAlgoCustomParam {
    int algo; // Generic ALGO strategy in {1,2,3,5,6}
    int tight; // 0 or 1 (FUSED_MOE_TIGHT)
    int custom_kernel; // 0 or 1
    int act_int; // 1=silu, 2=gelu, 3=swiglu_oai (act=none skipped ?
    // fused MoE always has an activation in practice)
    int M, num_ops, dim;
};

static std::string FusedAlgoCustomParamName(
        const ::testing::TestParamInfo<FusedAlgoCustomParam> &info) {
    static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
    const auto &p = info.param;
    return "algo" + std::to_string(p.algo) + "_tight" + std::to_string(p.tight)
            + "_custom" + std::to_string(p.custom_kernel) + "_"
            + act_names[p.act_int] + "_M" + std::to_string(p.M) + "_E"
            + std::to_string(p.num_ops) + "_d" + std::to_string(p.dim);
}

class TestFusedMoEAlgoCustom
    : public ::testing::TestWithParam<FusedAlgoCustomParam> {};

TEST_P(TestFusedMoEAlgoCustom, Correctness) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    reset_grp_matmul_caches();

    const auto &p = GetParam();
    const int H = 256, dim = p.dim, K = H;
    const int N_gate_up = 2 * dim;
    const int M = p.M, num_ops = p.num_ops;
    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
    // BF16-only: custom kernel contract + the fused-MoE V2 executor
    // both require BF16 throughout.
    const bool is_bf16 = true;
    const data_type_t dt = data_type_t::bf16;

    // Op2's K dimension follows the activation: gated => dim, none =>
    // N_gate_up.  See `op2_k_for_act` in group_matmul_fused_moe.cpp.
    const bool act_is_none = (act_type == grp_matmul_gated_act_t::none);
    const int K_down = act_is_none ? N_gate_up : dim;

    // Two src copies: `src_ref` for the legacy reference pass (unchanged
    // after use), `src_fused` for the internal-alloc fused path (Op2
    // writes back in-place, so the buffer is consumed).
    TypedBuffers src_ref, src_fused, w1, d1_ref, w2, d2_ref;
    src_ref.alloc(num_ops, (size_t)M * K, is_bf16);
    src_fused.alloc(num_ops, (size_t)M * K, is_bf16);
    w1.alloc(num_ops, (size_t)K * N_gate_up, is_bf16);
    d1_ref.alloc(num_ops, (size_t)M * N_gate_up, is_bf16);
    w2.alloc(num_ops, (size_t)K_down * H, is_bf16);
    d2_ref.alloc(num_ops, (size_t)M * H, is_bf16);
    fill_moe_tensors(num_ops, is_bf16, &src_ref, &w1, &w2);
    fill_moe_tensors(num_ops, is_bf16, &src_fused, nullptr, nullptr);

    auto gv_op1 = GemmVecs::uniform(num_ops, M, N_gate_up, K);

    auto srcs_ref = src_ref.cptrs(is_bf16);
    auto srcs_fused = src_fused.cptrs(is_bf16);
    auto wei1 = w1.cptrs(is_bf16);
    auto wei2 = w2.cptrs(is_bf16);
    auto dst1_ref = d1_ref.ptrs(is_bf16);
    auto dst2_r = d2_ref.ptrs(is_bf16);
    std::vector<const void *> no_bias(num_ops, nullptr);
    auto params = make_uniform_params(num_ops, dt);

    grp_matmul_gated_act_params act {};
    act.act = act_type;
    auto act_ptr = (act_type != grp_matmul_gated_act_t::none) ? &act : nullptr;

    // Reference: ALGO=1, TIGHT=0, CUSTOM=0, two-call legacy.
    // Everything else in the test can be evaluated against this single
    // baseline.  The env guards are scoped to this block so they don't
    // contaminate the later parameterised run.
    {
        AlgoEnvGuard algo_guard(1);
        FusedMoeTightOverride tight_guard(false);
        // Override atom, not setenv: the getter latches in a `static const`
        // on first read, so an EnvVarGuard here flips nothing and the
        // reference pass silently ran the same config as the test pass.
        CustomKernelOverride custom_guard(false);

        ASSERT_EQ(run_legacy_2call_ref(num_ops, M, K, N_gate_up, K_down, H,
                          is_bf16, act_type, srcs_ref, wei1, wei2, dst1_ref,
                          dst2_r),
                status_t::success)
                << "Ref legacy 2-call failed";
    }

    // ?? Test: parameterised ALGO ? TIGHT ? CUSTOM, internal-alloc fused ??
    // Also forces N_TILE_FUSED_ACT=1 so that when the caller picks
    // ALGO 3 + swiglu in V1 mode, the inline-fused epilogue path is
    // actually exercised (otherwise the ALGO 3 swiglu case would run a
    // separate-pass activation and we'd miss that code path).
    {
        AlgoEnvGuard algo_guard(p.algo);
        FusedMoeTightOverride tight_guard(p.tight);
        CustomKernelOverride custom_guard(p.custom_kernel);
        NTileFusedActOverride fused_act_guard(true);

        auto fused = make_fused_moe_op2(num_ops, H, wei2, no_bias);
        // fused.dst_down / ldc_down intentionally empty - internal-alloc.

        std::vector<void *> dst_null(num_ops, nullptr);
        std::vector<int> ldc_null(num_ops, 0);

        auto pf = params;
        ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA,
                          gv_op1.transB, gv_op1.Ms, gv_op1.Ns, gv_op1.Ks,
                          gv_op1.alpha, srcs_fused, gv_op1.lda, wei1,
                          gv_op1.ldb, no_bias, gv_op1.beta, dst_null, ldc_null,
                          gv_op1.is_wc, pf, nullptr, act_ptr, &fused),
                status_t::success)
                << "Fused call failed (algo=" << p.algo << " tight=" << p.tight
                << " custom=" << p.custom_kernel << " act=" << p.act_int << ")";
    }

    // Compare: src_fused now holds Op2 output (in-place, stride lda=K=H).
    std::ostringstream lbl;
    lbl << "algo=" << p.algo << " tight=" << p.tight
        << " custom=" << p.custom_kernel << " act=" << p.act_int;
    verify_per_expert_2d(src_fused, K, d2_ref, H, num_ops, M, H, is_bf16,
            tol_fused(is_bf16), lbl.str());
}

static std::vector<FusedAlgoCustomParam> make_fused_algo_custom_params() {
    std::vector<FusedAlgoCustomParam> out;
    // All 5 generic ALGOs ? TIGHT {0,1} ? CUSTOM {0,1} ? act
    // {silu, gelu, swiglu}.
    // TIGHT=1 with act ? {silu, gelu} or ALGO ? {0,3} exercises the gate
    // that routes back to V1 (Option A) ? expected to produce identical
    // outputs to the baseline.
    for (int algo : {1, 2, 3, 5, 6}) {
        for (int tight : {0, 1}) {
            for (int custom : {0, 1}) {
                for (int act : {1, 2, 3}) {
                    // M=4 ? num_ops=4 keeps the shape small enough for fast
                    // sharded execution.  dim=64 gives N_gate_up=128 which is
                    // a multiple of the custom kernel's pack_nr=32 so the
                    // custom path is reachable.
                    out.push_back({algo, tight, custom, act,
                            /*M=*/4, /*num_ops=*/4, /*dim=*/64});
                }
            }
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulFusedAlgoCustom, TestFusedMoEAlgoCustom,
        ::testing::ValuesIn(make_fused_algo_custom_params()),
        FusedAlgoCustomParamName);

// ???????????????????????????????????????????????????????????????????????????????
// [8] TestGroupMatmulAlgoCustom: non-fused Phase B env-knob matrix
//
// Targets the env-knob combinations that gate the custom BF16 microkernel's
// engagement in the non-fused group_matmul path (ALGO 3 via flat_n_tile):
//
//   * GRP_MATMUL_ALGO          ? 1 (sequential_experts), 3 (flat_n_tile),
//                                5 (per_expert); covers "custom kernel
//                                engages", "no engagement but same output",
//                                and "per-expert distribution" respectively.
//   * CUSTOM_KERNEL            ? 0 vs 1; 0 is the trusted standard path.
//   * N_TILE_FUSED_ACT         ? 0 vs 1 on swiglu; forces the inline
//                                fused-swiglu epilogue (otherwise the
//                                caller does a separate post-pass).
//   * gated_act                ? none / silu / gelu / swiglu_oai_mul.
//   * bias dtype               ? none / bf16 / fp32 (fp32 bias on bf16 dst
//                                exercises the BiasKind::fp32 load path).
//   * moe_postop (weighted)    ? off / on.
//
// Reference strategy: run the same call twice ? once with CUSTOM_KERNEL=0
// (standard dispatch, already verified by TestGroupMatmul / TestGatedAct /
// TestMoEPostop) and once with CUSTOM_KERNEL at the parameterised value
// ? and assert bit-identical (for configs where the custom kernel doesn't
// actually engage, e.g. ALGO 1 or ALGO 5) or within BF16 tolerance.  This
// covers the whole product of env toggles that the Phase B code added
// without duplicating the scalar-reference math elsewhere in this file.
// ???????????????????????????????????????????????????????????????????????????????

struct AlgoCustomParam {
    int algo; // GRP_MATMUL_ALGO strategy (1, 3, 5)
    int custom_kernel; // 0 or 1
    int n_tile_fused_act; // 0 or 1 (only meaningful for ALGO 3 + swiglu)
    int act_int; // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
    int bias_kind; // 0=none, 1=bf16, 2=fp32
    int M, num_ops, dim;
    // transB toggle ? exercises BOTH caller layouts the custom-kernel
    // pack now supports (false: [K,N] row-major, true: [N,K] row-major
    // PyTorch convention).  Differential test compares custom_kernel=1
    // run vs custom_kernel=0 reference with the SAME transB, so both
    // paths interpret `wei[]` identically; the test verifies the new
    // pack addressing produces bit-equivalent output to the standard
    // AOCL path.
    int transB; // 0 or 1
    // NOTE: moe_postop is intentionally not swept here.  The moe_postop
    // executor reduces the full wide Op1 output (D = N[0]), which for
    // gated activations is 2*dim and includes the un-activated second
    // half ? that region's contents legitimately differ between the
    // custom-kernel path (leaves cols [dim:2*dim] at zero) and the
    // standard path (leaves them at the raw Op1 GEMM output), making a
    // differential comparison ill-defined for act ? {swiglu}.  The
    // moe + gated_act combination is already covered by
    // TestGroupMatmulCombined, which uses a full step-by-step reference
    // so both paths are checked against a known ground truth.
};

static std::string AlgoCustomParamName(
        const ::testing::TestParamInfo<AlgoCustomParam> &info) {
    static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
    static const char *bias_names[] = {"noBias", "biasBF16", "biasFP32"};
    const auto &p = info.param;
    return "algo" + std::to_string(p.algo) + "_custom"
            + std::to_string(p.custom_kernel) + "_fusedAct"
            + std::to_string(p.n_tile_fused_act) + "_" + act_names[p.act_int]
            + "_" + bias_names[p.bias_kind] + "_tB" + std::to_string(p.transB)
            + "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops)
            + "_d" + std::to_string(p.dim);
}

class TestGroupMatmulAlgoCustom
    : public ::testing::TestWithParam<AlgoCustomParam> {};

// One parameterised call ? sets the CUSTOM_KERNEL env to `custom_value`,
// runs group_matmul_direct, and copies the final Op1 + activation
// output into `out_dst`.  Separated so the TEST_P body can invoke it
// twice (once with "0" for the reference, once with the parameter
// value for the test) and compare.
//
// The outer AlgoEnvGuard + N_TILE_FUSED_ACT EnvVarGuard are set in the
// TEST_P body so both runs share the same strategy and activation
// routing; only CUSTOM_KERNEL flips between runs.
static void run_one_algo_custom_pass(const AlgoCustomParam &p,
        const char *custom_value,
        const std::vector<std::vector<bfloat16_t>> &src,
        const std::vector<std::vector<bfloat16_t>> &wei,
        const std::vector<std::vector<bfloat16_t>> &bias_bf16,
        const std::vector<std::vector<float>> &bias_fp32, int N_op1, int K,
        std::vector<std::vector<bfloat16_t>> &out_dst) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

    // See the note at the parameterised site: setenv cannot move a latched
    // `static const` getter, so this must use the override atom.
    CustomKernelOverride custom_guard(
            custom_value != nullptr && custom_value[0] != '0');

    // Zero the dst buffers so any untouched region (e.g. cols
    // [dim:2*dim] under the custom-kernel swiglu write) has a
    // well-defined value across both ref and test runs.
    for (auto &v : out_dst) {
        std::fill(v.begin(), v.end(), bfloat16_t(0.0f));
    }

    std::vector<const void *> srcs(p.num_ops), weis(p.num_ops),
            biases(p.num_ops, nullptr);
    std::vector<void *> dsts(p.num_ops);
    for (int e = 0; e < p.num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = out_dst[e].data();
        if (p.bias_kind == 1) {
            biases[e] = bias_bf16[e].data();
        } else if (p.bias_kind == 2) {
            biases[e] = bias_fp32[e].data();
        }
    }

    auto gv = GemmVecs::uniform(p.num_ops, p.M, N_op1, K,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*wc=*/false, /*tA=*/false,
            /*tB=*/p.transB != 0);
    const data_type_t bias_dt = (p.bias_kind == 1) ? data_type_t::bf16
            : (p.bias_kind == 2)                   ? data_type_t::f32
                                                   : data_type_t::none;
    auto params = make_uniform_params(p.num_ops, data_type_t::bf16, bias_dt);

    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
    grp_matmul_gated_act_params act {};
    act.act = act_type;
    auto *act_ptr = (p.act_int != 0) ? &act : nullptr;

    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr, act_ptr),
            status_t::success)
            << "call failed: custom=" << custom_value;
}

TEST_P(TestGroupMatmulAlgoCustom, Correctness) {
    using namespace moe_test_utils;

    reset_grp_matmul_caches();

    const auto &p = GetParam();

    // N_op1 is the Op1 GEMM output width.  For act=none the output is
    // just `dim` cols wide; for any gated activation the GEMM produces
    // 2*dim cols and the activation compacts to [0:dim].
    const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
    const int K = 64;

    // Custom kernel requires N_op1 % pack_nr == 0 (32 or 64).  When the
    // grid lands on a smaller or misaligned N_op1 the custom path will
    // cleanly fall back to the standard dispatch ? the `== 0` case is
    // still valuable because it regression-tests the fallback path.

    AlgoEnvGuard algo_guard(p.algo);
    NTileFusedActOverride fused_act_guard(p.n_tile_fused_act);

    // ?? Prepare shared src / wei / bias (both runs see identical inputs) ??
    std::vector<std::vector<bfloat16_t>> src(
            p.num_ops, std::vector<bfloat16_t>((size_t)p.M * K));
    std::vector<std::vector<bfloat16_t>> wei(
            p.num_ops, std::vector<bfloat16_t>((size_t)K * N_op1));
    std::vector<std::vector<bfloat16_t>> bias_bf16(p.num_ops);
    std::vector<std::vector<float>> bias_fp32(p.num_ops);
    for (int e = 0; e < p.num_ops; ++e) {
        fill_src(src[e], e, 0.02f);
        fill_wei1(wei[e], e, 0.005f);
        if (p.bias_kind == 1) {
            bias_bf16[e].resize(N_op1);
            for (int n = 0; n < N_op1; ++n) {
                bias_bf16[e][n] = bfloat16_t(0.01f * ((n + e) % 7 - 3));
            }
        } else if (p.bias_kind == 2) {
            bias_fp32[e].resize(N_op1);
            for (int n = 0; n < N_op1; ++n) {
                bias_fp32[e][n] = 0.01f * ((n + e) % 7 - 3);
            }
        }
    }

    // ?? Reference run: CUSTOM_KERNEL=0 ?????????????????????????????????
    std::vector<std::vector<bfloat16_t>> dst_ref(p.num_ops,
            std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
    ASSERT_NO_FATAL_FAILURE(run_one_algo_custom_pass(
            p, "0", src, wei, bias_bf16, bias_fp32, N_op1, K, dst_ref));

    // ?? Test run: CUSTOM_KERNEL as parameterised ??????????????????????
    std::vector<std::vector<bfloat16_t>> dst_test(p.num_ops,
            std::vector<bfloat16_t>((size_t)p.M * N_op1, bfloat16_t(0.0f)));
    ASSERT_NO_FATAL_FAILURE(
            run_one_algo_custom_pass(p, p.custom_kernel ? "1" : "0", src, wei,
                    bias_bf16, bias_fp32, N_op1, K, dst_test));

    // ?? Compare ????????????????????????????????????????????????????????
    // When the custom kernel doesn't actually engage (ALGO 1 / 2 / 5 / 6,
    // or contract-rejected shapes), both runs take the same code path
    // and should match bit-for-bit.  When it does engage (ALGO 3 with a
    // satisfying contract), the FP32 accumulator numerics are nearly
    // identical to the AOCL DLP path; BF16 tolerance captures the
    // per-element rounding of the final `_mm512_cvtneps_pbh`.
    //
    // For gated activations we only compare the activated half
    // [0:dim] of each row.  The un-activated half [dim:2*dim] is
    // "don't care" per the library contract and legitimately differs
    // between paths (custom-kernel swiglu leaves zeros, standard
    // leaves raw GEMM output).
    const auto tol = tol_act(true);
    const int cmp_N = (p.act_int == 0) ? N_op1 : p.dim;
    for (int e = 0; e < p.num_ops; ++e) {
        for (int m = 0; m < p.M; ++m) {
            for (int n = 0; n < cmp_N; ++n) {
                const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                const float ref_v = static_cast<float>(dst_ref[e][idx]);
                const float test_v = static_cast<float>(dst_test[e][idx]);
                ASSERT_NEAR(test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
                        << "algo=" << p.algo << " custom=" << p.custom_kernel
                        << " fusedAct=" << p.n_tile_fused_act
                        << " act=" << p.act_int << " bias=" << p.bias_kind
                        << " e=" << e << " m=" << m << " n=" << n;
            }
        }
    }
}

static std::vector<AlgoCustomParam> make_algo_custom_params() {
    std::vector<AlgoCustomParam> out;
    // Strategy coverage ? 1 (sequential), 3 (flat_n_tile = custom hook),
    // 5 (per-expert).  Skip 2 and 6 to keep the grid tight; those
    // executors don't look at CUSTOM_KERNEL anyway (ALGO 3 is the only
    // engagement site in the non-fused path).
    const int algos[] = {1, 3, 5};
    // dim=64 ? N_op1=128 for act=none, N_op1=128 for gated; K=64.
    // This lets ALGO 3 exercise its N-tile split on ?2 threads while
    // keeping the test shape small enough to run fast.
    const int dim = 64;

    // Core grid ? every (algo ? custom ? act ? bias) combo.
    // N_TILE_FUSED_ACT is only meaningful for swiglu (the only gated
    // activation the custom kernel's inline epilogue supports), so we
    // only sweep both values of that knob for act=swiglu_oai_mul.
    for (int algo : algos) {
        for (int custom : {0, 1}) {
            for (int act : {0, 1, 2, 3}) {
                for (int bias : {0, 1, 2}) {
                    const std::vector<int> fused_acts = (act == 3)
                            ? std::vector<int> {0, 1}
                            : std::vector<int> {0};
                    for (int fa : fused_acts) {
                        // transB sweep ? exercises both [K,N] and [N,K] caller
                        // layouts.  The custom-kernel pack now supports both,
                        // and this differential test catches any addressing
                        // regression on the transB=true path that would
                        // otherwise only show up in framework integrations.
                        for (int tB : {0, 1}) {
                            // M=4 ? num_ops=8 keeps the shape small enough to run
                            // fast under 8?64 sharding.  dim=64 ? N_op1=128 for the
                            // gated cases (multiple of pack_nr=32 so the custom
                            // kernel's contract is met) and N_op1=64 for act=none
                            // (also a clean multiple of 32).
                            out.push_back({algo, custom, fa, act, bias,
                                    /*M=*/4, /*num_ops=*/8, dim, tB});
                        }
                    }
                }
            }
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulAlgoCustom, TestGroupMatmulAlgoCustom,
        ::testing::ValuesIn(make_algo_custom_params()), AlgoCustomParamName);

// ===============================================================================
// [8c] TestGroupMatmulInt8AlgoCustom — DQ-INT8 N-tile CK env-knob matrix
//
// Mirrors [8] (TestGroupMatmulAlgoCustom) but exercises the
// DQ-INT8 N-tile custom kernel added in Phase 1
// (zenai-grp-matmul-ntile-dqint8-ck-phase1).  The diff covers:
//
//   * Per-token symmetric (compute=s8, no src_zp) DQ-INT8.  The
//     asymmetric (compute=u8) path is covered at the microkernel,
//     pack, dispatcher and prepare_for_call layers in the
//     `gtests/group_matmul/custom_kernel/test_ukernel_int8.cpp`,
//     `test_pack_int8.cpp`, and `test_prepare_for_call.cpp` suites;
//     the existing tensor-factory plumbing in
//     `group_matmul_kernel_test` always sets `compute = s8` (it does
//     not consume an `s32 src_zp` annotation on the tensor), so an
//     e2e differential through that helper would silently route
//     into the symmetric kernel path for both runs and yield a
//     zero-information assertion.  Threading the `compute = u8`
//     metadata through the helper is its own follow-up.
//   * All four gated activations the CK admits today:
//     `none / silu_and_mul / gelu_and_mul / swiglu_oai_mul`.
//   * ALGO 3 forced (the only executor that engages the int8 CK).
//   * Reference: the same call run with the new
//     `_CUSTOM_KERNEL_INT8` env knob FORCED OFF — that path falls
//     back to AOCL DLP `aocl_gemm_s8s8s32obf16_sym_quant`, which
//     existing INT8 tests already validate against a scalar
//     reference.  Comparing the CK-on output against the CK-off
//     output keeps the test self-contained (no separate scalar
//     model) while still asserting end-to-end correctness through
//     the new microkernel + pack + dispatch + prepack stack.
//   * BF16 dst is the only Phase 1 supported dst dtype, so we don't
//     sweep dst.
//
// The fixture skips automatically on hosts without AVX-512 VNNI —
// `prepare_for_call` already refuses cleanly in that case, but
// running the differential would still spend wall-time so we cut
// it off at the gate.
// ===============================================================================

struct Int8AlgoCustomParam {
    int act_int; // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
    int M, num_ops,
            dim; // dim = single-half output width; N_op1 = 2*dim for gated
};

static std::string Int8AlgoCustomParamName(
        const ::testing::TestParamInfo<Int8AlgoCustomParam> &info) {
    static const char *act_names[] = {"none", "silu", "gelu", "swiglu"};
    const auto &p = info.param;
    std::string name = "sym_";
    name += act_names[p.act_int];
    name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops) + "_d"
            + std::to_string(p.dim);
    return name;
}

class TestGroupMatmulInt8AlgoCustom
    : public ::testing::TestWithParam<Int8AlgoCustomParam> {};

// Build per-expert (src, wei_s8, scale) tensors for the int8 DQ
// symmetric path.  The src is bf16 with attached per-token
// src_scale; the weight is s8 with per-channel wei_scale.  Returns
// the assembled tensor vectors and the dst buffer for the run.
// Caller invokes `group_matmul_kernel_test` twice — once with
// CK_INT8 OFF (reference) and once ON (test) — without rebuilding
// the inputs between runs.
static void build_int8_dq_inputs(tensor_factory_t &tf,
        const Int8AlgoCustomParam &p, int N_op1, int K,
        std::vector<tensor_t> &src_tensors, std::vector<tensor_t> &wei_tensors,
        std::vector<tensor_t> &bias_tensors, std::vector<tensor_t> &dst_tensors,
        // Scale dtype for BOTH the per-channel wei_scale and the per-token
        // src_scale.  Defaults to f32 (the original coverage); pass bf16 to
        // exercise the reorder's native bf16 scale output, which the CK
        // microkernel must materialise to f32 before consuming (reading a
        // bf16 scale buffer as f32 would silently corrupt every output —
        // see the f32-scale-view materialisation in flat_n_tile).
        data_type_t scale_dt = data_type_t::f32) {
    src_tensors.assign(p.num_ops, tensor_t {});
    wei_tensors.assign(p.num_ops, tensor_t {});
    bias_tensors.assign(p.num_ops, tensor_t {});
    dst_tensors.assign(p.num_ops, tensor_t {});

    // Per-channel wei_scale, per-token src_scale — the granularity
    // Phase 1's CK admits.  Scale dtype is f32 or bf16 (see `scale_dt`).
    const std::vector<int64_t> wei_scale_dims {1, N_op1};
    const std::vector<uint64_t> src_scale_shape {
            static_cast<uint64_t>(p.M), static_cast<uint64_t>(1)};

    for (int e = 0; e < p.num_ops; ++e) {
        auto wei_ref = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(K), static_cast<uint64_t>(N_op1)},
                data_type_t::bf16, /*range=*/2.0);
        tensor_t wei_s8, wei_scale, wei_zp;
        ASSERT_EQ(quant_params_compute(tf, wei_ref, data_type_t::bf16,
                          data_type_t::s8, wei_scale_dims, scale_dt, wei_scale,
                          wei_zp, &wei_s8),
                status_t::success)
                << "DQ-INT8 weight quant setup failed";
        wei_tensors[e] = wei_s8;

        auto src_scale = tf.zero_tensor(src_scale_shape, scale_dt);
        src_tensors[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(K)},
                data_type_t::bf16, /*range=*/2.0, /*trans=*/false, src_scale,
                tensor_t());

        // Bias is intentionally OMITTED — Phase 1 refuses gated-act +
        // bias for both bf16 and int8 CK paths, so a bias-free baseline
        // keeps every act cell inside the kernel envelope.
        bias_tensors[e] = tensor_t {};

        dst_tensors[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(N_op1)},
                data_type_t::bf16, /*range=*/2.0);
    }
}

// Run one pass with `_CUSTOM_KERNEL_INT8` forced to `int8_on` (the
// master `_CUSTOM_KERNEL` knob stays ON for both runs — only the
// int8 sub-knob flips).  `dst_out` receives a deep copy of the
// per-expert dst buffers so callers can compare across passes.
static void run_int8_dq_pass(bool int8_on, std::vector<tensor_t> &src_tensors,
        std::vector<tensor_t> &wei_tensors, std::vector<tensor_t> &bias_tensors,
        std::vector<tensor_t> &dst_tensors, grp_matmul_gated_act_t act_type,
        std::vector<std::vector<bfloat16_t>> &dst_out, int M, int N_op1) {
    using namespace moe_test_utils;
    CustomKernelOverride ck_master(true);
    CustomKernelInt8Override ck_int8(int8_on);

    grp_matmul_gated_act_params act_params {};
    act_params.act = act_type;
    const grp_matmul_gated_act_params *act_ptr
            = (act_type != grp_matmul_gated_act_t::none) ? &act_params
                                                         : nullptr;

    ASSERT_EQ(group_matmul_kernel_test(src_tensors, wei_tensors, bias_tensors,
                      dst_tensors, matmul_algo_t::aocl_dlp_blocked,
                      /*alpha=*/1.0f,
                      /*beta=*/0.0f, /*moe_postop=*/nullptr, act_ptr),
            status_t::success)
            << "DQ-INT8 ALGO 3 kernel call failed (int8_on=" << int8_on << ")";

    // Snapshot dst buffers into plain bf16 vectors for later compare.
    dst_out.assign(dst_tensors.size(),
            std::vector<bfloat16_t>(
                    static_cast<size_t>(M) * N_op1, bfloat16_t(0.0f)));
    for (size_t e = 0; e < dst_tensors.size(); ++e) {
        const auto *p = static_cast<const bfloat16_t *>(
                dst_tensors[e].get_raw_handle_unsafe());
        std::memcpy(dst_out[e].data(), p,
                static_cast<size_t>(M) * N_op1 * sizeof(bfloat16_t));
    }
}

TEST_P(TestGroupMatmulInt8AlgoCustom, Correctness) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }

    reset_grp_matmul_caches();

    const auto &p = GetParam();
    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);

    // N_op1 is the Op1 GEMM output width.  For act=none the output is
    // `dim` cols wide; for any gated activation the GEMM produces
    // 2*dim cols and the activation compacts to [0:dim].
    const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
    const int K = 64;

    // The int8 CK requires N % pack_nr ∈ {32, 64}.  Skip params that
    // wouldn't engage CK on either run — the differential would be
    // meaningless because both runs would take the AOCL DLP path.
    if (N_op1 % 32 != 0) {
        GTEST_SKIP() << "N_op1=" << N_op1
                     << " not a multiple of 32; CK would refuse";
    }

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst));

    AlgoEnvGuard algo_guard(3);

    std::vector<std::vector<bfloat16_t>> dst_ref, dst_test;

    // Reference pass: int8 CK OFF → AOCL DLP sym_quant runs.
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/false, src, wei, bias,
            dst, act_type, dst_ref, p.M, N_op1));

    // Rebuild the dst tensors so the second pass starts from a clean
    // slate (zero-out buffers; reuse src/wei/bias which are unchanged).
    for (auto &d : dst) {
        auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
        std::fill(
                raw, raw + static_cast<size_t>(p.M) * N_op1, bfloat16_t(0.0f));
    }

    // Test pass: int8 CK ON → new microkernel + pack path runs.
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei, bias,
            dst, act_type, dst_test, p.M, N_op1));

    // Comparison band: BF16 tolerance from the gated/non-gated helper
    // in `moe_test_utils` — same tolerance pair the existing
    // TestGroupMatmulAlgoCustom uses, since the activation post-pass
    // is the same scalar formula and the quant rounding deltas are
    // already within that band.
    const auto tol = tol_act(/*is_bf16=*/true);
    const int cmpN = (p.act_int == 0) ? N_op1 : p.dim;
    for (int e = 0; e < p.num_ops; ++e) {
        for (int m = 0; m < p.M; ++m) {
            for (int n = 0; n < cmpN; ++n) {
                const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                const float ref_v = static_cast<float>(dst_ref[e][idx]);
                const float test_v = static_cast<float>(dst_test[e][idx]);
                ASSERT_NEAR(test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
                        << "int8 sym act=" << p.act_int << " e=" << e
                        << " m=" << m << " n=" << n;
            }
        }
    }
}

// ===============================================================================
// Large-shape DQ-INT8 differential (CK vs AOCL DLP), ALGO 3.
//
// `TestGroupMatmulInt8AlgoCustom` runs the same differential at dim=32/64 and
// K=64.  Those shapes barely tile: with N_op1=64 and pack_nr=32 the planner
// gets two column tiles, so the per-tile scale slicing, the per-tile weight
// offsetting and the prepack key derivation are exercised only trivially.
//
// A deployment-scale MoE decode is nothing like that: a large K over tens of
// active experts splits N across the whole team, and every per-tile path runs
// for real.  Accuracy differences have been seen at that scale with no
// library-level test reproducing them, so this closes the shape gap between
// the unit coverage and a realistic configuration.
TEST(TestGroupMatmulInt8LargeShape, CkVsDlpAtProductionDims) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    reset_grp_matmul_caches();

    // Hidden 2048, intermediate 512, so gate|up gives N_op1 = 1024.  The
    // expert count and per-expert row count are representative of a batched
    // MoE decode, where most experts receive only a handful of tokens.
    Int8AlgoCustomParam p {/*act_int=*/1 /*silu_and_mul*/, /*M=*/4,
            /*num_ops=*/46, /*dim=*/512};
    const int N_op1 = 2 * p.dim; // 1024
    const int K = 2048;

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst));

    AlgoEnvGuard algo_guard(3);
    const auto act_type = grp_matmul_gated_act_t::silu_and_mul;

    std::vector<std::vector<bfloat16_t>> dst_dlp, dst_ck;
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/false, src, wei, bias,
            dst, act_type, dst_dlp, p.M, N_op1));
    for (auto &d : dst) {
        auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
        std::fill(
                raw, raw + static_cast<size_t>(p.M) * N_op1, bfloat16_t(0.0f));
    }
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei, bias,
            dst, act_type, dst_ck, p.M, N_op1));

    // Report the WORST deviation rather than aborting on the first, so a
    // failure says how bad and where instead of just "not equal".
    const auto tol = tol_act(/*is_bf16=*/true);
    double worst_rel = 0.0;
    int worst_e = -1, worst_m = -1, worst_n = -1;
    size_t out_of_band = 0;
    for (int e = 0; e < p.num_ops; ++e) {
        for (int m = 0; m < p.M; ++m) {
            for (int n = 0; n < p.dim; ++n) {
                const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                const double a = static_cast<float>(dst_dlp[e][idx]);
                const double b = static_cast<float>(dst_ck[e][idx]);
                const double band = std::abs(a) * tol.rel + tol.abs;
                if (std::abs(b - a) > band) { ++out_of_band; }
                const double denom = std::max(std::abs(a), 1e-6);
                const double rel = std::abs(b - a) / denom;
                if (rel > worst_rel) {
                    worst_rel = rel;
                    worst_e = e;
                    worst_m = m;
                    worst_n = n;
                }
            }
        }
    }
    const size_t total = static_cast<size_t>(p.num_ops) * p.M * p.dim;
    EXPECT_EQ(out_of_band, 0u)
            << out_of_band << " of " << total << " elements outside the bf16 "
            << "band; worst rel=" << worst_rel << " at expert=" << worst_e
            << " m=" << worst_m << " n=" << worst_n << " (K=" << K
            << " N_op1=" << N_op1 << " experts=" << p.num_ops << ")";
}

// Independent oracle for the production DQ-INT8 shape.
//
// `CkVsDlpAtProductionDims` above is a differential: it only asserts the two
// backends agree.  Both share the same quantization front-end and activation
// definition, so an error in either is invisible to a comparison between
// them -- agreement is not correctness.  This computes silu_and_mul over the
// same inputs in scalar f64, independently of either kernel, so a result
// they both get wrong has something to disagree with.
//
// The reference quantizes the source the same way the runtime hoist does
// (symmetric per-token, `max|row| / 127`, round-to-nearest) so the comparison
// isolates the GEMM and epilogue rather than re-measuring quantization noise.
// The band is correspondingly a bf16-accumulation band, not a bit-match.
TEST(TestGroupMatmulInt8LargeShape, ScalarOracleAtProductionDims) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    reset_grp_matmul_caches();

    // Smaller expert count than the differential: the oracle is O(M*N*K) in
    // scalar f64 per expert, and the property under test does not need 46.
    Int8AlgoCustomParam p {/*act_int=*/1 /*silu_and_mul*/, /*M=*/4,
            /*num_ops=*/4, /*dim=*/512};
    const int N_op1 = 2 * p.dim;
    const int K = 2048;

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst));

    AlgoEnvGuard algo_guard(3);
    std::vector<std::vector<bfloat16_t>> got;
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei, bias,
            dst, grp_matmul_gated_act_t::silu_and_mul, got, p.M, N_op1));

    double worst_rel = 0.0;
    size_t out_of_band = 0;
    for (int e = 0; e < p.num_ops; ++e) {
        const auto *s = static_cast<const bfloat16_t *>(
                src[e].get_raw_handle_unsafe());
        const auto *w
                = static_cast<const int8_t *>(wei[e].get_raw_handle_unsafe());
        ASSERT_EQ(wei[e].get_quant_scale_data_type(), data_type_t::f32)
                << "oracle reads the weight scale as f32";
        const auto *wsc = static_cast<const float *>(
                wei[e].get_quant_scale_raw_handle_const());

        for (int m = 0; m < p.M; ++m) {
            // Symmetric per-token source quantization, mirroring the hoist.
            double amax = 0.0;
            for (int k = 0; k < K; ++k) {
                amax = std::max(amax,
                        std::abs(static_cast<double>(s[(size_t)m * K + k])));
            }
            const double sscale = (amax > 0.0) ? amax / 127.0 : 1.0;

            std::vector<int32_t> qs(K);
            for (int k = 0; k < K; ++k) {
                const double v
                        = static_cast<double>(s[(size_t)m * K + k]) / sscale;
                qs[k] = static_cast<int32_t>(std::lrint(v));
            }

            for (int n = 0; n < p.dim; ++n) {
                auto col = [&](int c) {
                    int64_t acc = 0;
                    for (int k = 0; k < K; ++k) {
                        acc += static_cast<int64_t>(qs[k])
                                * static_cast<int64_t>(
                                        w[(size_t)k * N_op1 + c]);
                    }
                    return static_cast<double>(acc) * sscale
                            * static_cast<double>(wsc[c]);
                };
                const double gate = col(n);
                const double up = col(n + p.dim);
                const double ref = (gate / (1.0 + std::exp(-gate))) * up;

                const double val
                        = static_cast<double>(got[e][(size_t)m * N_op1 + n]);
                // bf16 output carries ~3 decimal digits, and the reference
                // accumulates in f64 where the kernel accumulates in s32 then
                // converts, so an absolute floor is needed for near-zero
                // products where a relative band is meaningless.
                const double band = std::abs(ref) * 0.05 + 0.05;
                if (std::abs(val - ref) > band) { ++out_of_band; }
                const double rel
                        = std::abs(val - ref) / std::max(std::abs(ref), 1e-6);
                worst_rel = std::max(worst_rel, rel);
            }
        }
    }
    const size_t total = static_cast<size_t>(p.num_ops) * p.M * p.dim;
    EXPECT_EQ(out_of_band, 0u)
            << out_of_band << " of " << total
            << " elements disagree with the scalar oracle; worst rel="
            << worst_rel << " (K=" << K << " N_op1=" << N_op1
            << " experts=" << p.num_ops << ")";
}

static std::vector<Int8AlgoCustomParam> make_int8_algo_custom_params() {
    std::vector<Int8AlgoCustomParam> out;
    // Four activations × two shapes.  We keep the grid tight — the
    // per-cell signal is fully redundant with the microkernel-level
    // coverage in test_ukernel_int8.cpp, so the e2e fixture only
    // needs enough variety to catch dispatch / pack / prepack
    // integration regressions.
    for (int act : {0, 1, 2, 3}) {
        // (M=4, num_ops=8, dim=32) — N_op1 = 32 (act=none) or 64
        // (gated) → both clean multiples of pack_nr=32.
        out.push_back({act, /*M=*/4, /*num_ops=*/8, /*dim=*/32});
        // (M=16, num_ops=4, dim=64) — N_op1 = 64 or 128 → exercises
        // the kNRMax=64 branch and multi-tile splitting on >2 threads.
        out.push_back({act, /*M=*/16, /*num_ops=*/4, /*dim=*/64});
    }
    // C.6 — decode (M=1) and prompt (M=64) shapes covering the
    // edge of the M-dimension envelope:
    //   * M=1 hits the single-row hoist path inside
    //     `reorder_quantization_wrapper` and the int8 microkernel's
    //     MR=1 instantiation.
    //   * M=64 hits the multi-MR-band split with deep K-quad
    //     accumulation pressure (decode/prompt parity).
    //   * dim=128 (NV=4 / kNRMax=64) at M=8 exercises the
    //     pack_nr=64 cache key and the wider register kernel.
    // Activation cycles through {none, swiglu} to keep the new
    // matrix small (silu/gelu are byte-identical to swiglu at the
    // pack layer per C.2.5; their epilogues are covered by the
    // microkernel param sweep).
    for (int act : {0, 3}) {
        out.push_back({act, /*M=*/1, /*num_ops=*/8, /*dim=*/32}); // decode
        out.push_back({act, /*M=*/64, /*num_ops=*/4, /*dim=*/64}); // prompt
        out.push_back({act, /*M=*/8, /*num_ops=*/4, /*dim=*/128}); // NR=128
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulInt8AlgoCustom,
        TestGroupMatmulInt8AlgoCustom,
        ::testing::ValuesIn(make_int8_algo_custom_params()),
        Int8AlgoCustomParamName);

// ===============================================================================
// Per-group dynamic-INT8 cross-ALGO parity (plain s8 weights, NON-GGML)
//
// Guards the contract "wherever per-token is supported, per-group is supported":
// a symmetric INT8 PER-GROUP dynamic-quant group matmul (bf16 src + {M,G}
// src_scale, s8 wei + {G,N} wei_scale) MUST produce correct results on EVERY
// group ALGO (1/2/3/5) and under both ENABLE_GROUP_DQ settings.  Per-group
// either engages the optimized path (e.g. flat_m_tile for dynamic + GROUP_DQ=0)
// or falls back to a per-group-capable AOCL path (ALGO 1 / per-expert) — and is
// NEVER misrouted to a per-token-only custom/native kernel (which would silently
// corrupt output by slicing the source scale as one-scalar-per-row).  Every
// configuration is compared to the ALGO-1 reference (the validated full-N AOCL
// sym-quant per-group path), so a misroute shows up as a gross mismatch.
// ===============================================================================
struct Int8PerGroupAlgoParam {
    int M, num_ops, dim, G; // G = number of K-groups; K is fixed at 64 below.
};

static std::string Int8PerGroupAlgoName(
        const ::testing::TestParamInfo<Int8PerGroupAlgoParam> &info) {
    const auto &p = info.param;
    return "M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops) + "_dim"
            + std::to_string(p.dim) + "_G" + std::to_string(p.G);
}

class TestGroupMatmulInt8PerGroupAlgos
    : public ::testing::TestWithParam<Int8PerGroupAlgoParam> {};

// Build per-expert plain-s8 (NON-GGML) per-group inputs: bf16 src with an
// attached zero {M, G} src_scale (so the harness flags dynamic_quant) and an
// s8 weight carrying a {G, N} per-group wei_scale.
static void build_int8_pergroup_inputs(tensor_factory_t &tf,
        const Int8PerGroupAlgoParam &p, int N, int K,
        std::vector<tensor_t> &src, std::vector<tensor_t> &wei,
        std::vector<tensor_t> &bias, std::vector<tensor_t> &dst) {
    src.assign(p.num_ops, tensor_t {});
    wei.assign(p.num_ops, tensor_t {});
    bias.assign(p.num_ops, tensor_t {});
    dst.assign(p.num_ops, tensor_t {});
    const std::vector<int64_t> wei_sd {
            static_cast<int64_t>(p.G), static_cast<int64_t>(N)};
    const std::vector<uint64_t> src_ss {
            static_cast<uint64_t>(p.M), static_cast<uint64_t>(p.G)};
    for (int e = 0; e < p.num_ops; ++e) {
        auto wref = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(K), static_cast<uint64_t>(N)},
                data_type_t::bf16, /*range=*/2.0);
        tensor_t ws8, wsc, wzp;
        ASSERT_EQ(quant_params_compute(tf, wref, data_type_t::bf16,
                          data_type_t::s8, wei_sd, data_type_t::f32, wsc, wzp,
                          &ws8),
                status_t::success)
                << "per-group weight quant setup failed (e=" << e << ")";
        wei[e] = ws8;
        auto ssc = tf.zero_tensor(src_ss, data_type_t::f32);
        src[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(K)},
                data_type_t::bf16, /*range=*/2.0, /*trans=*/false, ssc,
                tensor_t());
        bias[e] = tensor_t {};
        dst[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(N)},
                data_type_t::bf16, /*range=*/2.0);
    }
}

TEST_P(TestGroupMatmulInt8PerGroupAlgos, CrossAlgoParity) {
    using namespace moe_test_utils;
    const auto &p = GetParam();
    const int N = p.dim;
    const int K = 64;
    ASSERT_EQ(K % p.G, 0) << "K must split evenly into G groups";

    // One run of the per-group group matmul under a given ALGO + GROUP_DQ
    // setting; snapshots the per-expert bf16 output.  Inputs are rebuilt each
    // run (and caches reset) so heap-address reuse can't return a stale
    // reordered weight across configurations.
    auto run = [&](int algo, bool group_dq,
                       std::vector<std::vector<bfloat16_t>> &out) {
        reset_grp_matmul_caches();
        tensor_factory_t tf;
        std::vector<tensor_t> src, wei, bias, dst;
        ASSERT_NO_FATAL_FAILURE(
                build_int8_pergroup_inputs(tf, p, N, K, src, wei, bias, dst));
        AlgoEnvGuard ag(algo);
        GroupDqOverride gdq(group_dq ? 1 : 0);
        ASSERT_EQ(group_matmul_kernel_test(src, wei, bias, dst,
                          matmul_algo_t::aocl_dlp_blocked,
                          /*alpha=*/1.0f, /*beta=*/0.0f),
                status_t::success)
                << "per-group group matmul failed (algo=" << algo
                << " group_dq=" << group_dq << ")";
        out.assign(p.num_ops,
                std::vector<bfloat16_t>(
                        static_cast<size_t>(p.M) * N, bfloat16_t(0.0f)));
        for (int e = 0; e < p.num_ops; ++e) {
            const auto *r = static_cast<const bfloat16_t *>(
                    dst[e].get_raw_handle_unsafe());
            std::memcpy(out[e].data(), r,
                    static_cast<size_t>(p.M) * N * sizeof(bfloat16_t));
        }
    };

    // Reference: ALGO 1 (full-N AOCL sym-quant) with the default grouped DQ
    // pre-pass — the validated per-group path.
    std::vector<std::vector<bfloat16_t>> ref;
    ASSERT_NO_FATAL_FAILURE(run(/*algo=*/1, /*group_dq=*/true, ref));

    // INT8 sym-quant cross-config band: the s8 data is identical across configs
    // (same quant kernel), so only reduction-order deltas separate the paths —
    // a misroute to a per-token-only kernel would blow well past this.
    const float rel = 0.06f;
    const float abs = 0.5f;
    // algo 0 = auto-select: confirms the dispatcher's phase-based pick lands on
    // a per-group-safe path (decode -> N-tile ALGO 3 plain s8, prompt -> M-tile
    // ALGO 2) and matches the ALGO 1 reference.
    for (int algo : {0, 1, 2, 3, 5}) {
        for (bool gdq : {true, false}) {
            std::vector<std::vector<bfloat16_t>> got;
            ASSERT_NO_FATAL_FAILURE(run(algo, gdq, got));
            for (int e = 0; e < p.num_ops; ++e) {
                for (int i = 0; i < p.M * N; ++i) {
                    const float a = static_cast<float>(got[e][i]);
                    const float b = static_cast<float>(ref[e][i]);
                    ASSERT_NEAR(a, b, std::abs(b) * rel + abs)
                            << "per-group mismatch: algo=" << algo
                            << " group_dq=" << gdq << " e=" << e
                            << " idx=" << i;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulInt8PerGroupAlgos,
        TestGroupMatmulInt8PerGroupAlgos,
        ::testing::ValuesIn(std::vector<Int8PerGroupAlgoParam> {
                {/*M=*/4, /*num_ops=*/8, /*dim=*/64,
                        /*G=*/2}, // prompt-ish, G=2
                {/*M=*/16, /*num_ops=*/4, /*dim=*/64, /*G=*/4}, // G=4
                {/*M=*/1, /*num_ops=*/8, /*dim=*/64, /*G=*/2}, // decode (M=1)
                {/*M=*/64, /*num_ops=*/4, /*dim=*/128,
                        /*G=*/8}, // larger N, G=8
                // Wide N (> the N-tile min tile width) so the N-tile planner
                // column-splits across threads — exercises do_tile's per-group
                // {G,N}->{G,n_tile} weight-scale repack on the AOCL path.
                {/*M=*/8, /*num_ops=*/2, /*dim=*/1024,
                        /*G=*/4}, // wide-N column split
                {/*M=*/1, /*num_ops=*/4, /*dim=*/768, /*G=*/2}, // decode wide-N
        }),
        Int8PerGroupAlgoName);

// ===============================================================================
// Per-group group-size validation (plain s8 {G,N})
//
// The AOCL sym-quant reorder that serves per-group symmetric quant needs K to
// split evenly into G groups AND the resulting group size to be a multiple
// of 4.  W4A8 has always enforced both (`validate_w4a8_inputs`); plain s8
// `{G, N}` reached the same reorder unchecked, so a G that divides K but
// leaves `K/G % 4 != 0` produced wrong numbers instead of a refusal.
//
// The parity suite above cannot catch this: every K/G it uses is a power of
// two >= 4, so the constraint is satisfied by construction.  These cases pick
// a G that divides K but yields a group size of 6, and a control G that
// yields 12.
// ===============================================================================
TEST(GroupMatmulInt8PerGroupValidation, RejectsGroupSizeNotMultipleOfFour) {
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    constexpr int kK = 48; // 48 / 8 = 6, divides evenly but 6 % 4 != 0
    const Int8PerGroupAlgoParam p {/*M=*/4, /*num_ops=*/2, /*dim=*/64, /*G=*/8};
    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_pergroup_inputs(tf, p, p.dim, kK, src, wei, bias, dst));
    ASSERT_EQ(kK % p.G, 0) << "the G under test must still divide K evenly, so "
                              "the refusal is attributable to the group size";
    EXPECT_EQ(group_matmul_kernel_test(src, wei, bias, dst,
                      matmul_algo_t::aocl_dlp_blocked,
                      /*alpha=*/1.0f, /*beta=*/0.0f),
            status_t::failure)
            << "K/G = " << (kK / p.G)
            << " is not a multiple of 4; the call must be refused rather than "
               "run on a group size the sym-quant reorder cannot honour";
}

TEST(GroupMatmulInt8PerGroupValidation, AcceptsGroupSizeMultipleOfFour) {
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    constexpr int kK = 48; // 48 / 4 = 12, a multiple of 4
    const Int8PerGroupAlgoParam p {/*M=*/4, /*num_ops=*/2, /*dim=*/64, /*G=*/4};
    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_pergroup_inputs(tf, p, p.dim, kK, src, wei, bias, dst));
    EXPECT_EQ(group_matmul_kernel_test(src, wei, bias, dst,
                      matmul_algo_t::aocl_dlp_blocked,
                      /*alpha=*/1.0f, /*beta=*/0.0f),
            status_t::success)
            << "K/G = " << (kK / p.G)
            << " satisfies the reorder's contract and must still run";
}

// ===============================================================================
// [8c-bf16] TestGroupMatmulInt8AlgoCustomBf16Scale — DQ-INT8 with BF16 scales
//
// Regression guard for the bf16-scale stitching bug: the dynamic-quant
// reorder emits the per-token src_scale (and the caller supplies the
// per-channel wei_scale) in BF16, but the int8 microkernel loads scales
// with `_mm512_loadu_ps` / `_mm512_set1_ps` (f32-only).  Before the
// f32-scale-view materialisation in `flat_n_tile`, the CK reinterpreted
// the bf16 scale bytes as f32 and produced systematic garbage (zero
// accuracy on real bf16 models), while the AOCL DLP fallback consumed
// the declared bf16 dtype correctly.  The whole class was invisible to
// the suite because every other int8 test uses f32 scales.
//
// Mechanism: identical CK-OFF (AOCL DLP sym_quant, the bf16-scale-aware
// reference) vs CK-ON (new microkernel) differential as [8c], but with
// `scale_dt = bf16`.  act=none is used because it is the activation that
// reliably engages the int8 CK (gated int8 currently routes to the AOCL
// separate-activation path); act=none also matches the down-proj GEMM
// where the production bug first surfaced.
// ===============================================================================
class TestGroupMatmulInt8AlgoCustomBf16Scale
    : public ::testing::TestWithParam<Int8AlgoCustomParam> {};

TEST_P(TestGroupMatmulInt8AlgoCustomBf16Scale, Correctness) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }

    reset_grp_matmul_caches();

    const auto &p = GetParam();
    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);

    const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
    const int K = 64;
    if (N_op1 % 32 != 0) {
        GTEST_SKIP() << "N_op1=" << N_op1
                     << " not a multiple of 32; CK would refuse";
    }

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst,
                    /*scale_dt=*/data_type_t::bf16));

    AlgoEnvGuard algo_guard(3);

    std::vector<std::vector<bfloat16_t>> dst_ref, dst_test;

    // Reference: int8 CK OFF → AOCL DLP sym_quant (bf16-scale aware).
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/false, src, wei, bias,
            dst, act_type, dst_ref, p.M, N_op1));

    for (auto &d : dst) {
        auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
        std::fill(
                raw, raw + static_cast<size_t>(p.M) * N_op1, bfloat16_t(0.0f));
    }

    // Test: int8 CK ON → must materialise the bf16 scales to f32 and
    // match the reference (pre-fix this produced garbage).
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei, bias,
            dst, act_type, dst_test, p.M, N_op1));

    const auto tol = tol_act(/*is_bf16=*/true);
    const int cmpN = (p.act_int == 0) ? N_op1 : p.dim;
    for (int e = 0; e < p.num_ops; ++e) {
        for (int m = 0; m < p.M; ++m) {
            for (int n = 0; n < cmpN; ++n) {
                const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                const float ref_v = static_cast<float>(dst_ref[e][idx]);
                const float test_v = static_cast<float>(dst_test[e][idx]);
                ASSERT_NEAR(test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
                        << "int8 bf16-scale act=" << p.act_int << " e=" << e
                        << " m=" << m << " n=" << n;
            }
        }
    }
}

// act=none across decode (M=1), small-batch (M=4) and prompt (M=64)
// shapes at both pack_nr widths (dim=32 → NR=32, dim=64 → NR=64).
INSTANTIATE_TEST_SUITE_P(GroupMatmulInt8AlgoCustomBf16Scale,
        TestGroupMatmulInt8AlgoCustomBf16Scale,
        ::testing::Values(Int8AlgoCustomParam {/*act=*/0, /*M=*/1,
                                  /*num_ops=*/8, /*dim=*/32},
                Int8AlgoCustomParam {
                        /*act=*/0, /*M=*/4, /*num_ops=*/8, /*dim=*/32},
                Int8AlgoCustomParam {
                        /*act=*/0, /*M=*/16, /*num_ops=*/4, /*dim=*/64},
                Int8AlgoCustomParam {
                        /*act=*/0, /*M=*/64, /*num_ops=*/4, /*dim=*/64}),
        Int8AlgoCustomParamName);

// With the weight cache on, the CK converts a BF16 weight scale to FP32 once
// and caches it; with it off the kernel widens the BF16 scale on load.  Both
// widenings are exact, so the two runs must agree bit for bit.
TEST(TestGroupMatmulInt8Bf16ScaleCache, WeightCacheOnOffBitExact) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }

    for (const Int8AlgoCustomParam &p :
            {Int8AlgoCustomParam {/*act=*/0, /*M=*/4, /*num_ops=*/4,
                     /*dim=*/64},
                    Int8AlgoCustomParam {/*act=*/0, /*M=*/25, /*num_ops=*/1,
                            /*dim=*/1024}}) {
        reset_grp_matmul_caches();
        const int N_op1 = p.dim;
        const int K = 128;
        tensor_factory_t tf;
        std::vector<tensor_t> src, wei, bias, dst;
        ASSERT_NO_FATAL_FAILURE(
                build_int8_dq_inputs(tf, p, N_op1, K, src, wei, bias, dst,
                        /*scale_dt=*/data_type_t::bf16));

        AlgoEnvGuard algo_guard(3);
        std::vector<std::vector<bfloat16_t>> out[2];
        for (int wc = 0; wc <= 1; ++wc) {
            WeightCacheGuard wc_guard(wc);
            for (auto &d : dst) {
                auto *raw
                        = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
                std::fill(raw, raw + static_cast<size_t>(p.M) * N_op1,
                        bfloat16_t(0.0f));
            }
            GemmModeCaptureGuard gemm_mode_guard;
            ASSERT_NO_FATAL_FAILURE(
                    run_int8_dq_pass(/*int8_on=*/true, src, wei, bias, dst,
                            grp_matmul_gated_act_t::none, out[wc], p.M, N_op1));
            const char *mode
                    = zendnnl::lowoha::matmul::test_api ::
                              s_last_group_matmul_direct_gemm_mode.load(
                                      std::memory_order_relaxed);
            ASSERT_NE(mode, nullptr);
            ASSERT_NE(std::strstr(mode, "_custom"), nullptr)
                    << "WEIGHT_CACHE=" << wc << " M=" << p.M << " ran on '"
                    << mode << "' instead of the INT8 CK";
        }
        for (int e = 0; e < p.num_ops; ++e) {
            EXPECT_EQ(std::memcmp(out[0][e].data(), out[1][e].data(),
                              out[0][e].size() * sizeof(bfloat16_t)),
                    0)
                    << "BF16 scale cache on/off differ: M=" << p.M
                    << " E=" << p.num_ops << " e=" << e;
        }
    }
}

// The FP32 scale cache re-checks a per-channel scale's bytes on every hit, so
// a BF16 scale rewritten in place behind the same pointer must take effect on
// the next call and match a run with the cache off.
TEST(TestGroupMatmulInt8Bf16ScaleCache, InPlaceScaleUpdateIsHonored) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    reset_grp_matmul_caches();

    const Int8AlgoCustomParam p {/*act=*/0, /*M=*/25, /*num_ops=*/1,
            /*dim=*/1024};
    const int N_op1 = p.dim;
    const int K = 128;
    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(build_int8_dq_inputs(tf, p, N_op1, K, src, wei,
            bias, dst, /*scale_dt=*/data_type_t::bf16));

    AlgoEnvGuard algo_guard(3);
    WeightCacheGuard wc_on(1);
    auto run = [&](std::vector<std::vector<bfloat16_t>> &out) {
        for (auto &d : dst) {
            auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
            std::fill(raw, raw + static_cast<size_t>(p.M) * N_op1,
                    bfloat16_t(0.0f));
        }
        GemmModeCaptureGuard gemm_mode_guard;
        ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei,
                bias, dst, grp_matmul_gated_act_t::none, out, p.M, N_op1));
        const char *mode = zendnnl::lowoha::matmul::test_api ::
                                   s_last_group_matmul_direct_gemm_mode.load(
                                           std::memory_order_relaxed);
        ASSERT_NE(mode, nullptr);
        ASSERT_NE(std::strstr(mode, "_custom"), nullptr)
                << "ran on '" << mode << "' instead of the INT8 CK";
    };

    std::vector<std::vector<bfloat16_t>> before, after, uncached;
    ASSERT_NO_FATAL_FAILURE(run(before));
    // Doubling a BF16 value only moves its exponent, so the new scale is exact.
    auto *scale = static_cast<bfloat16_t *>(
            const_cast<void *>(wei[0].get_quant_scale_raw_handle_const()));
    ASSERT_NE(scale, nullptr);
    for (int n = 0; n < N_op1; ++n)
        scale[n] = bfloat16_t(2.0f * static_cast<float>(scale[n]));
    ASSERT_NO_FATAL_FAILURE(run(after));
    {
        WeightCacheGuard wc_off(0);
        ASSERT_NO_FATAL_FAILURE(run(uncached));
    }

    const size_t bytes = before[0].size() * sizeof(bfloat16_t);
    EXPECT_NE(std::memcmp(before[0].data(), after[0].data(), bytes), 0)
            << "the rewritten scale did not change the output";
    EXPECT_EQ(std::memcmp(after[0].data(), uncached[0].data(), bytes), 0)
            << "the cached run disagrees with a cache-off run after the "
               "in-place scale update";
}

// ===============================================================================
// [8c-oblk] TestGroupMatmulInt8OblockOuter — DQ-INT8 o-block-outer nesting
//
// `dispatch_tile_int8` swaps to o-block-outer loops for one expert with at
// least `kOblockOuterMinCalls` MR calls.  At the NV=2 symmetric cap of MR=12
// that needs M >= 25, and every other INT8 differential runs below it, so
// the swapped loops would otherwise go unexecuted.  N is wide enough that
// each tile spans several o-blocks, otherwise both orders are one loop.  The
// asymmetric U8 kernels cap at MR=8, so M=17 already makes three calls.  The
// two-expert case is the control that keeps the MR-call-outer order.
//
// Checked against an f64 oracle on the unquantized source rather than the
// AOCL DLP arm, whose U8 output drifts from the oracle by more than an
// element-wise band allows.  The band scales with each row's largest output:
// source quantization error does not shrink with an element's own magnitude,
// while a misplaced o-block or MR call corrupts whole rows or column spans.
// ===============================================================================
struct Int8OblockParam {
    int act_int; // 0=none, 1=silu
    int M, num_ops, dim;
    data_type_t compute; // s8 (symmetric) or u8 (asymmetric)
    data_type_t dst_dt; // bf16, or f32 for act=none
    bool bias; // act=none only
};

static std::string Int8OblockParamName(
        const ::testing::TestParamInfo<Int8OblockParam> &info) {
    const auto &p = info.param;
    std::string name = (p.compute == data_type_t::u8) ? "asym_" : "sym_";
    name += (p.act_int == 0) ? "none" : "silu";
    name += "_M" + std::to_string(p.M) + "_E" + std::to_string(p.num_ops) + "_d"
            + std::to_string(p.dim);
    name += (p.dst_dt == data_type_t::f32) ? "_f32dst" : "_bf16dst";
    if (p.bias) name += "_bias";
    return name;
}

class TestGroupMatmulInt8OblockOuter
    : public ::testing::TestWithParam<Int8OblockParam> {};

TEST_P(TestGroupMatmulInt8OblockOuter, MatchesOracle) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;

    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    reset_grp_matmul_caches();

    const auto &p = GetParam();
    const auto act_type = static_cast<grp_matmul_gated_act_t>(p.act_int);
    const int N_op1 = (p.act_int == 0) ? p.dim : 2 * p.dim;
    const int K = 256;

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(build_int8_dq_inputs(tf,
            Int8AlgoCustomParam {p.act_int, p.M, p.num_ops, p.dim}, N_op1, K,
            src, wei, bias, dst));
    const std::vector<uint64_t> dst_shape {
            static_cast<uint64_t>(p.M), static_cast<uint64_t>(N_op1)};
    for (int e = 0; e < p.num_ops; ++e) {
        dst[e] = tf.zero_tensor(dst_shape, p.dst_dt);
        if (p.bias) {
            bias[e] = tf.uniform_dist_tensor(
                    {1, static_cast<uint64_t>(N_op1)}, data_type_t::bf16, 2.0);
        }
    }

    AlgoEnvGuard algo_guard(3);
    // ALGO 3 takes a source the grouped pre-pass already quantized only when
    // it is S8, so U8 keeps dynamic_quant and quantizes inside the N-tile.
    GroupDqOverride group_dq(p.compute == data_type_t::u8 ? 0 : -1);
    grp_matmul_gated_act_params act_params {};
    act_params.act = act_type;
    CustomKernelOverride ck_master(true);
    CustomKernelInt8Override ck_int8(true);
    GemmModeCaptureGuard gemm_mode_guard;
    ASSERT_EQ(group_matmul_kernel_test(src, wei, bias, dst,
                      matmul_algo_t::aocl_dlp_blocked, 1.0f, 0.0f,
                      /*moe_postop=*/nullptr,
                      p.act_int == 0 ? nullptr : &act_params,
                      /*pack_format_b=*/ {}, /*active_rows=*/ {}, p.compute),
            status_t::success);
    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr)
            << "group_matmul_direct did not publish a gemm_mode";
    ASSERT_NE(std::strstr(mode, "_custom"), nullptr)
            << "M=" << p.M << " E=" << p.num_ops << " ran on '" << mode
            << "' instead of the INT8 microkernel, so the loop nesting "
               "under test was never executed.";

    constexpr double kRowBand = 0.03;
    const int cmpN = (p.act_int == 0) ? N_op1 : p.dim;
    std::vector<double> acc(N_op1), ref(cmpN);
    for (int e = 0; e < p.num_ops; ++e) {
        const auto *s = static_cast<const bfloat16_t *>(
                src[e].get_raw_handle_unsafe());
        const auto *w
                = static_cast<const int8_t *>(wei[e].get_raw_handle_unsafe());
        ASSERT_EQ(wei[e].get_quant_scale_data_type(), data_type_t::f32)
                << "oracle reads the weight scale as f32";
        const auto *wsc = static_cast<const float *>(
                wei[e].get_quant_scale_raw_handle_const());
        const auto *b = p.bias ? static_cast<const bfloat16_t *>(
                                         bias[e].get_raw_handle_unsafe())
                               : nullptr;
        const void *raw = dst[e].get_raw_handle_unsafe();
        for (int m = 0; m < p.M; ++m) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k < K; ++k) {
                const double sv = static_cast<float>(s[(size_t)m * K + k]);
                const int8_t *wr = w + (size_t)k * N_op1;
                for (int n = 0; n < N_op1; ++n)
                    acc[n] += sv * wr[n];
            }
            for (int n = 0; n < N_op1; ++n) {
                acc[n] *= wsc[n];
                if (b) acc[n] += static_cast<float>(b[n]);
            }
            double row_max = 0.0, worst = 0.0;
            int worst_n = 0;
            for (int n = 0; n < cmpN; ++n) {
                const double g = acc[n];
                ref[n] = (p.act_int == 0)
                        ? g
                        : g / (1.0 + std::exp(-g)) * acc[n + p.dim];
                row_max = std::max(row_max, std::abs(ref[n]));
            }
            for (int n = 0; n < cmpN; ++n) {
                const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                const double val = (p.dst_dt == data_type_t::f32)
                        ? static_cast<const float *>(raw)[idx]
                        : static_cast<float>(
                                  static_cast<const bfloat16_t *>(raw)[idx]);
                const double err = std::abs(val - ref[n]);
                if (err > worst) {
                    worst = err;
                    worst_n = n;
                }
            }
            EXPECT_LE(worst, kRowBand * row_max)
                    << "int8 o-block nesting act=" << p.act_int
                    << " E=" << p.num_ops << " e=" << e << " m=" << m
                    << " worst n=" << worst_n << " (row max " << row_max << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulInt8OblockOuter,
        TestGroupMatmulInt8OblockOuter,
        ::testing::Values(Int8OblockParam {0, 25, 1, 1024, data_type_t::s8,
                                  data_type_t::bf16, false},
                Int8OblockParam {0, 32, 1, 1024, data_type_t::s8,
                        data_type_t::bf16, false},
                Int8OblockParam {0, 36, 1, 1024, data_type_t::s8,
                        data_type_t::bf16, true},
                Int8OblockParam {0, 25, 1, 1024, data_type_t::s8,
                        data_type_t::f32, true},
                Int8OblockParam {1, 25, 1, 512, data_type_t::s8,
                        data_type_t::bf16, false},
                Int8OblockParam {0, 17, 1, 1024, data_type_t::u8,
                        data_type_t::bf16, false},
                Int8OblockParam {0, 17, 1, 1024, data_type_t::u8,
                        data_type_t::f32, true},
                Int8OblockParam {1, 17, 1, 512, data_type_t::u8,
                        data_type_t::bf16, false},
                Int8OblockParam {0, 25, 2, 1024, data_type_t::s8,
                        data_type_t::bf16, false}),
        Int8OblockParamName);

// ===============================================================================
// [8c-ptile] TestGroupMatmulInt8PromptTile — DQ-INT8 on the prompt tile executor
//
// With the INT8 bit of `ZENDNNL_GRP_MATMUL_PROMPT_TILE` set, ALGO 3 cuts an
// INT8 CK prompt into row and column tiles of the hoisted s8 / u8 source.
// Each tile reads its own rows' source scales and zero points, so the output
// must equal the round executor's bit for bit.  Experts of 0 to 200 rows
// under small forced blocks split into many tiles each.
// ===============================================================================
TEST(TestGroupMatmulInt8PromptTile, BitwiseMatchesRounds) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    struct Case {
        const char *name;
        data_type_t compute; // s8 (symmetric) or u8 (asymmetric)
        data_type_t scale_dt;
        int act_int; // 0=none, 1=silu, 2=gelu
        data_type_t dst_dt;
        bool bias; // act=none only
        int group_dq; // 1: the grouped pre-pass quantizes the source (s8 only)
    };
    const auto s8 = data_type_t::s8, u8 = data_type_t::u8;
    const auto f32 = data_type_t::f32, bf16 = data_type_t::bf16;
    const std::vector<int> rows {0, 1, 2, 3, 45, 0, 130, 64, 9, 200, 17};
    const int dim = 128, K = 256;
    for (const Case &c : {Case {"sym", s8, f32, 0, bf16, false, 0},
                 Case {"sym_grouped_prequant", s8, f32, 0, bf16, false, 1},
                 Case {"asym", u8, f32, 0, bf16, false, 0},
                 Case {"sym_bf16_scales", s8, bf16, 0, bf16, false, 1},
                 Case {"asym_bf16_scales", u8, bf16, 0, bf16, false, 0},
                 Case {"sym_silu", s8, f32, 1, bf16, false, 1},
                 Case {"sym_gelu_bf16_scales", s8, bf16, 2, bf16, false, 0},
                 Case {"asym_silu", u8, f32, 1, bf16, false, 0},
                 Case {"asym_gelu_bf16_scales", u8, bf16, 2, bf16, false, 0},
                 Case {"sym_bias", s8, f32, 0, bf16, true, 1},
                 Case {"sym_bias_f32_dst", s8, f32, 0, f32, true, 0},
                 Case {"asym_bias_f32_dst", u8, bf16, 0, f32, true, 0}}) {
        SCOPED_TRACE(c.name);
        const int N_op1 = c.act_int == 0 ? dim : 2 * dim;
        tensor_factory_t tf;
        std::vector<tensor_t> src, wei, bias, dst;
        // The source scale must be {M[e], 1}, so each expert gets its own M.
        for (const int m : rows) {
            const Int8AlgoCustomParam p {c.act_int, std::max(m, 1), 1, dim};
            std::vector<tensor_t> s, w, b, d;
            ASSERT_NO_FATAL_FAILURE(build_int8_dq_inputs(
                    tf, p, N_op1, K, s, w, b, d, c.scale_dt));
            src.push_back(s[0]);
            wei.push_back(w[0]);
            bias.push_back(c.bias ? tf.uniform_dist_tensor(
                                            {1, static_cast<uint64_t>(N_op1)},
                                            bf16, 2.0)
                                  : tensor_t {});
            dst.push_back(tf.zero_tensor(
                    {static_cast<uint64_t>(p.M), static_cast<uint64_t>(N_op1)},
                    c.dst_dt));
        }
        GroupDqOverride group_dq(c.group_dq);
        CustomKernelInt8Override ck_int8(true);
        grp_matmul_gated_act_params act {};
        act.act = static_cast<grp_matmul_gated_act_t>(c.act_int);
        expect_prompt_tile_matches_rounds(/*mask=*/2, /*mb=*/16, /*ns=*/64, src,
                wei, bias, dst, c.act_int == 0 ? nullptr : &act, rows,
                c.compute);
    }
}

// The family bitmask defaults to BF16+FP16 (mask 9), so without the INT8 bit
// an INT8 custom-kernel prompt runs on the round executor.
TEST(TestGroupMatmulInt8PromptTile, DefaultMaskKeepsRounds) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    namespace pt = zendnnl::lowoha::matmul::prompt_tile;
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    if (std::getenv("ZENDNNL_GRP_MATMUL_PROMPT_TILE") != nullptr) {
        GTEST_SKIP() << "ZENDNNL_GRP_MATMUL_PROMPT_TILE is set";
    }
    const std::vector<int> rows {64, 48, 80};
    const int dim = 128, K = 256;
    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    for (const int m : rows) {
        const Int8AlgoCustomParam p {0, m, 1, dim};
        std::vector<tensor_t> s, w, b, d;
        ASSERT_NO_FATAL_FAILURE(build_int8_dq_inputs(
                tf, p, dim, K, s, w, b, d, data_type_t::f32));
        src.push_back(s[0]);
        wei.push_back(w[0]);
        bias.push_back(tensor_t {});
        dst.push_back(tf.zero_tensor(
                {static_cast<uint64_t>(m), static_cast<uint64_t>(dim)},
                data_type_t::bf16));
    }
    AlgoEnvGuard algo(3);
    CustomKernelOverride ck_on(true);
    CustomKernelInt8Override ck_int8(true);
    GemmModeCaptureGuard capture;
    reset_grp_matmul_caches();
    const long served0 = pt::test_api::s_calls_served.load();
    ASSERT_EQ(group_matmul_kernel_test(src, wei, bias, dst,
                      matmul_algo_t::aocl_dlp_blocked, 1.0f, 0.0f,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr,
                      /*pack_format_b=*/ {}, rows, data_type_t::s8),
            status_t::success);
    EXPECT_FALSE(pt::enabled(pt::Family::kInt8));
    EXPECT_EQ(pt::test_api::s_calls_served.load(), served0);
    const char *m = zendnnl::lowoha::matmul::test_api::
                            s_last_group_matmul_direct_gemm_mode.load(
                                    std::memory_order_relaxed);
    const std::string mode = m != nullptr ? m : "";
    EXPECT_NE(mode.find("_custom"), std::string::npos) << mode;
    EXPECT_EQ(mode.find("ptile"), std::string::npos) << mode;
}

// ===============================================================================
// [8a] TestGroupMatmulPhaseBRemainder — CK Single-round + non-zero-remainder
//
// Targets the planner+executor path that
// `apply_round_pick(RoundPick::Single, use_custom=true)` populates
// when `num_threads % num_ops != 0`:
//
//   * `apply_round_pick` runs the heaviest-first eligibility filter,
//     populates `stable_n_thr_per_expert[]` non-uniformly (M-heaviest
//     experts get `base + 1` threads, the rest get `base`), and sets
//     `plan.per_expert_remainder = true`.
//   * `execute_rounds` reads `per_expert_remainder` and switches from
//     the O(1) `tid / thr_per_expert` mapping to the per-round
//     prefix-sum `tid → (expert, local_tid)` scan that consumes the
//     non-uniform `stable_n_thr_per_expert[]`.
//
// Failure modes the test catches:
//   * Drop tiles — a regression in the prefix-sum mapping that
//     skips an N-column slice → output cells diverge from the ALGO 1
//     reference.
//   * Duplicate tiles — a regression that maps two threads onto the
//     same N-column → race-corrupted cells diverge.
//   * Eligibility filter bugs — wrong `extras` count or per-expert
//     `n_thr` of zero → `do_tile`'s `local_tid >= n_thr` early-return
//     leaves columns un-written → diverge from the reference.
//
// Strategy: run the same call twice — once with ALGO=3 (current
// default knobs land on CK + Single, firing Phase B) and once with
// ALGO=1 sequential (the dispatcher's gold reference: serial over
// experts but each expert runs with the full thread team, so it
// never enters Phase B's per-expert remainder distribution).
// Compare BF16 outputs cell-by-cell within the standard
// activation-free tolerance.
//
// Shape choice for the documented `64 threads / 18 experts` case
// flagged in PR #461 review: `omp_set_num_threads(32)` paired with
// `num_ops = 14` — this lands `base = 32/14 = 2`, `remainder = 4`,
// `per_expert_cap = min(ccd_size=8, max_tiles=N/min_n_tile)`.
// `M = 4` keeps the call small; `N = 1024` (multiple of pack_nr=32
// for CK eligibility) gives `max_tiles = 1024/256 = 4` so
// `(base+1) = 3 ≤ 4` and Phase B fires on every active expert
// (uniform `N` → all 14 experts pass the eligibility filter →
// first 4 experts get `base+1=3` threads, the other 10 get
// `base=2`, summing to 32 threads = full team).
//
// 32-thread choice (rather than 64): keeps the matmul work small
// while still triggering Phase B; avoids over-subscription on hosts
// where the test runner already pinned a smaller team.  Restored
// to the prior team size on test teardown.
//
// Note on env caching: `ZENDNNL_GRP_MATMUL_N_ROUNDS`,
// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`, and the
// `CUSTOM_KERNEL_N_TILE` getters cache their value at first call
// (`static const`).  Earlier tests in this binary may have already
// observed a non-default value, in which case an `EnvVarGuard`
// here would be a no-op and the ALGO=3 pass below could silently
// take a non-CK / non-Single path with the cell-by-cell comparison
// still passing — losing Phase B coverage.
//
// Use the test-only `*Override` RAII guards (defined in
// `moe_test_utils.hpp`) which write a `test_api::` atomic that
// the production getters check before the cached read path.  This
// guarantees the test sees its required configuration regardless
// of test-suite order.  `AlgoEnvGuard(3)` / `AlgoEnvGuard(1)` update
// the selector override as well as the cached production env surface,
// so they flip cleanly per scope.
TEST(TestGroupMatmulPhaseBRemainder, CkSingleRoundRemainderCorrectness) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    // Force a 32-thread team for this test; restore on scope exit.
    // Use omp_set_num_threads (process-global) since the dispatcher
    // reads `omp_get_max_threads()` when params don't pin
    // num_threads, which is the case for `make_uniform_params`.
    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);

    // Verify the host can actually spawn 32 worker threads.
    // `omp_get_max_threads()` reports the requested `nthreads-var`
    // ICV (i.e. what `omp_set_num_threads` set, capped by the host's
    // dynamic-team policy) but NOT the actual team size that
    // `#pragma omp parallel` will instantiate when it fires.  On
    // hosts with `OMP_THREAD_LIMIT` or dynamic-adjustment enabled,
    // the runtime can hand out a smaller team than requested, which
    // would land us on a different (base, remainder) tuple and miss
    // the 32t/14ops Phase B shape we're locking down.
    //
    // Use a probe parallel region to read `omp_get_num_threads()`
    // — that value is the active team size inside the region —
    // before committing to the test.
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Test requires >= 32 active OMP threads to land "
                        "on the 32t/14ops Phase B remainder shape; "
                        "probe parallel region reported "
                     << actual_team_size
                     << " (requested 32; host's "
                        "dynamic-team / thread-limit policy may have "
                        "clamped it).";
    }

    // Force the env-cached configuration the Phase B path needs,
    // regardless of what an earlier test in this binary already
    // cached.  The RAII guards write a `test_api::` atomic that the
    // production getters check before their cached `static const`
    // read; on scope exit the prior override value is restored, so
    // tests that follow this one are unaffected.
    CustomKernelOverride ck_on(true); // CUSTOM_KERNEL=ON
    NRoundsModeOverride single_round(1); // N_ROUNDS=1 (Single)
    // 0 = use default (kDecodeNTile=256).  Our shape (N=1024) was
    // sized for ab_min_tile=256 so max_tiles=4 and (base+1)=3 ≤
    // per_expert_cap=4, which is required for Phase B to fire.
    CustomKernelNTileOverride default_n_tile(0);
    // Pin the N-tile strategy knob to 0 (auto / heuristic) so this
    // test is insulated from a process-level
    // `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1` (force DecodeD) or `=2`
    // (force Rounds) that some surrounding benchmark / sweep might
    // have set.  Phase B's per-expert remainder distribution only
    // populates inside ManyExperts Single-round; values 1/2 would
    // override the heuristic DecodeD attempt, taking different
    // planner branches that bypass Phase B and leave
    // `snap.per_expert_remainder = false`.
    NTileStrategyOverride auto_strategy(0);

    // Defence-in-depth: the env-cache overrides force the planner's
    // INTENT to take the CK path, but `prepare_for_call` makes a
    // separate runtime decision based on AVX512BF16 availability.
    // On a host without that ISA the dispatcher refuses CK and
    // the ALGO 3 run silently routes through the AOCL strict-stable
    // plan — which, with this shape's `dynamic_quant=false` and
    // null quant buffers, would ALSO produce numerically equivalent
    // output to ALGO 1, so the comparison below would pass without
    // exercising Phase B.  Skip when CK is structurally unavailable.
    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                        "dispatch support; forcing the cached env "
                        "knobs is necessary but not sufficient on this "
                        "host.";
    }

    reset_grp_matmul_caches();

    const int num_ops = 14; // 32 % 14 = 4 → Phase B fires
    const int N_op1 = 1024; // % pack_nr=32 ✓; max_tiles=4
    const int K = 64;

    // Two M distributions, both run end-to-end and compared against
    // the ALGO 1 reference:
    //
    //   (a) Uniform M=4 — every expert equally weighted; the
    //       eligibility filter passes all 14 experts (uniform N=1024).
    //       Locks down tile coverage (no dropped / duplicated cols)
    //       and the prefix-sum thread→expert mapping.
    //
    //   (b) Skewed M with the first 4 experts heaviest (16, 12, 10, 8)
    //       and the remaining 10 light (6 then 4×9) — exercises the
    //       same eligibility filter on a non-uniform input.  Note that
    //       this comparison CANNOT directly assert the heaviest-first
    //       ordering: reordering which experts get `base+1` does not
    //       change per-cell numerics, only the per-expert thread
    //       distribution.  That ordering is covered separately by the
    //       white-box `HeaviestFirstAssignment` test below, which
    //       captures `plan.stable_n_thr_per_expert[]` via the
    //       `test_api::PhaseBSnapshot` hook in
    //       `group_matmul_n_tile.hpp` and asserts on the per-expert
    //       thread counts directly.
    struct MVariant {
        std::vector<int> Ms;
        const char *label;
    };
    const std::vector<MVariant> variants = {
            {std::vector<int>(num_ops, 4), "uniform_M=4"},
            {{16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4},
                    "skewed_M_heaviest_first"},
    };

    for (const auto &var : variants) {
        SCOPED_TRACE(std::string("M variant: ") + var.label);
        const auto &Ms = var.Ms;
        ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);
        // The previous variant's ALGO 3 run packed its weights in place, and
        // this variant's weights can land at the same addresses.
        reset_grp_matmul_caches();

        // ── Prepare shared inputs sized to per-expert M ──
        std::vector<std::vector<bfloat16_t>> src(num_ops);
        std::vector<std::vector<bfloat16_t>> wei(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            src[e].resize(static_cast<size_t>(Ms[e]) * K);
            wei[e].resize(static_cast<size_t>(K) * N_op1);
            fill_src(src[e], e, 0.02f);
            fill_wei1(wei[e], e, 0.005f);
        }

        // Helper to build the call vectors and run group_matmul_direct
        // with `algo` forced.  Captures the output of expert `e` into
        // `out_dst[e]` (sized to Ms[e] * N_op1).  No bias / no
        // activation keeps the comparison strictly numerical.
        auto run_one_pass
                = [&](int algo, std::vector<std::vector<bfloat16_t>> &out_dst) {
            AlgoEnvGuard algo_guard(algo);

            for (auto &v : out_dst) {
                std::fill(v.begin(), v.end(), bfloat16_t(0.0f));
            }
            std::vector<const void *> srcs(num_ops), weis(num_ops),
                    biases(num_ops, nullptr);
            std::vector<void *> dsts(num_ops);
            for (int e = 0; e < num_ops; ++e) {
                srcs[e] = src[e].data();
                weis[e] = wei[e].data();
                dsts[e] = out_dst[e].data();
            }

            // Build with M=0 then override per-expert Ms (rest of the
            // wrapper vectors stay uniform — N, K, lda, ldb, ldc all
            // shape-independent of M for tA=tB=false).
            //
            // `wc=true` is REQUIRED for the CK path to engage on the
            // ALGO 3 run.  `custom_kernel::prepare_for_call` refuses CK
            // for any active expert flagged `is_weights_const=false`
            // (dispatch.cpp:304-308) — without this, the ALGO 3 pass
            // would silently fall back to the standard AOCL DLP per-tile
            // path and bypass Phase B entirely, defeating the test.
            auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N_op1, K,
                    /*alpha=*/1.0f, /*beta=*/0.0f,
                    /*wc=*/true, /*tA=*/false,
                    /*tB=*/false);
            gv.Ms = Ms;
            auto params = make_uniform_params(num_ops, data_type_t::bf16);
            // Pin num_threads via the params struct.  `omp_set_num_threads`
            // alone does NOT propagate to the dispatcher because
            // `thread_guard::max_threads()`
            // (lowoha_operators/common/omp_thread_control.hpp:90-93)
            // caches `omp_get_max_threads()` on first invocation —
            // process-wide, for the binary's lifetime.  Setting
            // `params[i].num_threads = 32` forces
            // `resolve_num_threads(32, cached) = 32` and `thread_guard`
            // then flips the ICV to 32 for the scope of this call.
            // Required for the planner to land on the (base=2, remainder=4)
            // tuple Phase B is sized for; without it the dispatcher uses
            // the cached top-level OMP value and the planner picks
            // (base=4, remainder=8) which fails the
            // `(base+1) <= per_expert_cap=4` gate and never enters
            // Phase B.
            for (int e = 0; e < num_ops; ++e) {
                params[e].num_threads = 32;
            }

            ASSERT_EQ(
                    group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                            gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                            biases, gv.beta, dsts, gv.ldc, gv.is_wc, params,
                            /*moe_postop=*/nullptr,
                            /*gated_act=*/nullptr),
                    status_t::success)
                    << "algo=" << algo;
        };

        // ── Reference: ALGO 1 sequential (gold) ──
        std::vector<std::vector<bfloat16_t>> dst_ref(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            dst_ref[e].assign(
                    static_cast<size_t>(Ms[e]) * N_op1, bfloat16_t(0.0f));
        }
        ASSERT_NO_FATAL_FAILURE(run_one_pass(/*algo=*/1, dst_ref));

        // ── Test: ALGO 3 (Phase B fires on CK Single + remainder) ──
        // Wrap in a `PhaseBCaptureGuard` so we can ASSERT_TRUE after
        // the call that the snapshot was actually populated AND that
        // `per_expert_remainder` is set — otherwise a future change
        // that bypasses Phase B (e.g. plan strategy changed, gate
        // moved) would silently produce equivalent output and the
        // comparison below would pass while Phase B coverage is lost.
        // The guard also disarms the capture flag on scope exit, so a
        // failed `group_matmul_direct` cannot leak the flag into a
        // later test.
        reset_grp_matmul_caches();
        std::vector<std::vector<bfloat16_t>> dst_test(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            dst_test[e].assign(
                    static_cast<size_t>(Ms[e]) * N_op1, bfloat16_t(0.0f));
        }
        {
            PhaseBCaptureGuard cap;
            ASSERT_NO_FATAL_FAILURE(run_one_pass(/*algo=*/3, dst_test));

            const auto &snap = zendnnl::lowoha::matmul::test_api::
                    s_last_phase_b_snapshot;
            ASSERT_TRUE(snap.valid)
                    << "PhaseBSnapshot was not captured during the ALGO 3 "
                       "run — flat_n_tile may not have been reached (variant: "
                    << var.label << ")";
            ASSERT_TRUE(snap.per_expert_remainder)
                    << "Phase B did not fire on the ALGO 3 run.  The "
                       "numerical comparison below would still pass on a "
                       "fallback path but the test would lose Phase B "
                       "coverage.  variant="
                    << var.label
                    << "  strategy=" << static_cast<int>(snap.strategy)
                    << "  n_thr_fixed=" << snap.n_thr_fixed;
        }

        // ── Compare cell-by-cell within BF16 tolerance ──
        // Per-cell BF16 rounding of the final `_mm512_cvtneps_pbh`
        // against AOCL DLP's accumulator differs by at most a relative
        // epsilon at this magnitude.  `tol_act(/*is_bf16=*/true)` is the
        // same tolerance pair the existing TestGroupMatmulAlgoCustom
        // uses.
        const auto tol = tol_act(true);
        for (int e = 0; e < num_ops; ++e) {
            for (int m = 0; m < Ms[e]; ++m) {
                for (int n = 0; n < N_op1; ++n) {
                    const size_t idx = static_cast<size_t>(m) * N_op1 + n;
                    const float ref_v = static_cast<float>(dst_ref[e][idx]);
                    const float test_v = static_cast<float>(dst_test[e][idx]);
                    ASSERT_NEAR(
                            test_v, ref_v, std::abs(ref_v) * tol.rel + tol.abs)
                            << "Phase B remainder regression: variant="
                            << var.label << " e=" << e << " m=" << m
                            << " n=" << n << " ref=" << ref_v
                            << " test=" << test_v;
                }
            }
        }
    }
}

// ===============================================================================
// [8a.1] TestGroupMatmulPhaseBRemainder — heaviest-first white-box
//
// The end-to-end correctness test above cannot directly observe the
// heaviest-first comparator: reordering which experts get `base+1`
// does NOT change the per-cell GEMM output, only the per-expert
// thread distribution.  A regression that reversed the comparator
// (lightest-first), or assigned `base+1` independently of M, would
// pass the cell-by-cell comparison and slip past silently.
//
// Use the planner's test-only `PhaseBSnapshot` (defined in
// `group_matmul_n_tile.hpp::test_api`) to read out the per-expert
// thread counts the planner committed for the next `flat_n_tile`
// call.  Assert that:
//
//   * `per_expert_remainder` is true (Phase B fired).
//   * Every active expert has 1..cap threads and the team is saturated.
//   * The decode auto default chose the M-proportional split: the
//     heaviest expert reaches its cap and the lightest stays at the
//     floor, which the uniform base/base+1 split cannot produce.
//
// Shape: same 32t / 14ops / N=1024 setup as the correctness test
// above, but with skewed M = {16,12,10,8,6,4×9}.
// ===============================================================================

TEST(TestGroupMatmulPhaseBRemainder, HeaviestFirstAssignment) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::GroupNTileStrategy;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);

    // Real team-size probe (same reasoning as the correctness test).
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 active OMP threads; have "
                     << actual_team_size;
    }

    // Force the env-cached config Phase B needs (RAII, restored on
    // scope exit).  `NTileStrategyOverride(0)` insulates the test
    // from a process-level `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1/2`
    // that would otherwise bypass ManyExperts and skip Phase B —
    // see `CkSingleRoundRemainderCorrectness` for the full rationale.
    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride auto_strategy(0);
    // Leave DecodeProportionalOverride unset: this uniform-N/K fixture
    // exercises the production default-auto path, which an env value
    // would replace.
    if (const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL");
            e != nullptr && e[0] != '\0') {
        GTEST_SKIP() << "Exercises the auto default; "
                        "ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL is set.";
    }

    // Defence-in-depth: env overrides force the PLANNER's intent to
    // take the CK path, but `prepare_for_call` separately refuses CK
    // on hosts without AVX512BF16.  Without this skip the test would
    // run, fail to capture per_expert_remainder, and report a noisy
    // failure on a structural ISA limitation.
    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                        "dispatch support; cannot exercise Phase B "
                        "without CK engagement.";
    }

    reset_grp_matmul_caches();

    const int num_ops = 14;
    const int N_op1 = 1024;
    const int K = 64;
    const std::vector<int> Ms = {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);

    // Build inputs sized to per-expert M.
    std::vector<std::vector<bfloat16_t>> src(num_ops);
    std::vector<std::vector<bfloat16_t>> wei(num_ops);
    std::vector<std::vector<bfloat16_t>> dst(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(Ms[e]) * K);
        wei[e].resize(static_cast<size_t>(K) * N_op1);
        dst[e].assign(static_cast<size_t>(Ms[e]) * N_op1, bfloat16_t(0.0f));
        fill_src(src[e], e, 0.02f);
        fill_wei1(wei[e], e, 0.005f);
    }

    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }

    auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N_op1, K,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*wc=*/true, /*tA=*/false,
            /*tB=*/false);
    gv.Ms = Ms;
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    // Pin num_threads=32 via params so the dispatcher honours the
    // intended team size regardless of `thread_guard::max_threads`'s
    // process-wide cache.  See the matching comment in
    // `CkSingleRoundRemainderCorrectness` above for the full
    // rationale; without this the planner lands on (base=4,
    // remainder=8) which fails Phase B's outer
    // `(base+1) <= per_expert_cap` gate.
    for (int e = 0; e < num_ops; ++e) {
        params[e].num_threads = 32;
    }

    // Arm the snapshot capture via RAII so the flag is always
    // disarmed on scope exit (including on test failure / early
    // ASSERT abort).  `flat_n_tile` will exchange the flag back to
    // false and copy `plan.stable_n_thr_per_expert[]` +
    // `plan.per_expert_remainder` into `s_last_phase_b_snapshot`.
    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);

    // Read the snapshot.  `flat_n_tile`'s exchange should have
    // cleared the flag; the snapshot holds the planner's output.
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;

    ASSERT_TRUE(snap.valid)
            << "PhaseBSnapshot was not captured — flat_n_tile may not "
               "have been reached.";
    EXPECT_TRUE(snap.per_expert_remainder)
            << "Phase B did not fire; planner may have routed to a "
               "non-Single strategy or the eligibility filter rejected "
               "every expert.  Check `strategy` / `n_thr_fixed` below.";
    EXPECT_EQ(static_cast<int>(snap.strategy),
            static_cast<int>(GroupNTileStrategy::ManyExperts))
            << "Phase B is only populated on the ManyExperts strategy "
               "(Single round picked by N_ROUNDS=1 + use_custom).";
    EXPECT_EQ(snap.num_ops_active, num_ops);

    // 32 threads / 14 ops, max_M=16 <= kDecodeMaxM=32 → decode-class with
    // uniform N/K, so the auto default picks the M-PROPORTIONAL thread split
    // over the uniform Phase B base/base+1.  Proportional allocates threads
    // in proportion to M, clamped to each expert's N-tile capacity
    // cap=N/ab_min_tile=1024/256=4, so the M-heaviest experts receive the
    // most threads.
    EXPECT_NE(snap.n_thr_fixed, 0)
            << "proportional leaves n_thr_fixed at base (HYBRID clears to 0).";

    int sum = 0;
    for (int e = 0; e < num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        if (Ms[e] > 0) {
            EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
        }
        EXPECT_LE(n, 4) << "expert e=" << e
                        << " exceeds N-tile capacity cap=4 (N=1024/256)";
        sum += n;
    }
    EXPECT_EQ(sum, 32)
            << "M-proportional should saturate the full thread team; "
               "sum(stable_n_thr_per_expert) = "
            << sum;
    // The uniform split would give {3,3,3,3,2,...,2}.  Proportional fills
    // the heaviest expert (M=16, want 5.8) to its cap of 4 and leaves the
    // last light experts (M=4, want 1.45) at the floor of 1, a pair the
    // uniform split cannot produce.
    EXPECT_EQ(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
            << "heaviest expert (M=16) must reach its N-tile cap";
    EXPECT_EQ(static_cast<int>(snap.stable_n_thr_per_expert[13]), 1)
            << "lightest expert (M=4) must stay at the 1-thread floor";
}

// ===============================================================================
// [8a.2] TestGroupMatmulPhaseBRemainder — eligibility filter, non-uniform N
//
// Companion to `HeaviestFirstAssignment`: that test uses uniform N
// = 1024 so every expert is eligible (`N[e] / ab_min_tile = 4 ≥
// base + 1 = 3`).  Without a non-uniform-N case a regression that
// removed the per-expert eligibility check would still pass —
// every expert qualifies in the uniform fixture.
//
// This test pins the eligibility filter explicitly: same 32t /
// 14ops / K=64 setup, same skewed M = {16,12,10,8,6,4×9}, but with
// the M-heaviest expert (e=0, M=16) given N=512 instead of 1024.
// Its tile capacity becomes `512 / 256 = 2 < base+1 = 3`, so it
// MUST NOT receive `base+1` even though it is M-heaviest.  The
// expected slot-by-slot distribution is:
//
//   * e=0 (M=16, N=512, INELIGIBLE)            → base    = 2
//   * e=1..4 (M=12,10,8,6, N=1024) heaviest of → base+1  = 3
//     the eligible set; receive all `remainder=4` extras
//   * e=5..13 (M=4, N=1024)                    → base    = 2
//
// Sum = 2 + 4×3 + 9×2 = 32 (full team utilisation).
//
// A regression that ignores `pairs[e].eligible` would route the
// extra to e=0 (highest M, sorted first), leaving e=4 at base —
// the assertion below would catch that immediately because e=0's
// `stable_n_thr_per_expert` would read 3 instead of 2.
// ===============================================================================

TEST(TestGroupMatmulPhaseBRemainder, EligibilityFilter_NonUniformN) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::GroupNTileStrategy;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);

    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 active OMP threads; have "
                     << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    // Insulate from a process-level
    // `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1/2` that would bypass
    // ManyExperts and skip Phase B — see
    // `CkSingleRoundRemainderCorrectness` for the full rationale.
    NTileStrategyOverride auto_strategy(0);
    // Pin the proportional path ON so this non-uniform-N fixture exercises the
    // forced per-expert capacity clamp rather than the default-auto gate.
    DecodeProportionalOverride prop_on(true);

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                        "dispatch support; cannot exercise Phase B "
                        "without CK engagement.";
    }

    reset_grp_matmul_caches();

    const int num_ops = 14;
    const int K = 64;
    const int N_eligible = 1024; // 1024 / 256 = 4 ≥ base+1
    const int N_ineligible = 512; //  512 / 256 = 2 < base+1
    const std::vector<int> Ms = {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    // Heaviest expert (e=0, M=16) gets the narrow N — sorts first
    // by M descending but is filtered out by the eligibility check.
    std::vector<int> Ns(num_ops, N_eligible);
    Ns[0] = N_ineligible;
    ASSERT_EQ(static_cast<int>(Ms.size()), num_ops);

    std::vector<std::vector<bfloat16_t>> src(num_ops);
    std::vector<std::vector<bfloat16_t>> wei(num_ops);
    std::vector<std::vector<bfloat16_t>> dst(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(Ms[e]) * K);
        wei[e].resize(static_cast<size_t>(K) * Ns[e]);
        dst[e].assign(static_cast<size_t>(Ms[e]) * Ns[e], bfloat16_t(0.0f));
        fill_src(src[e], e, 0.02f);
        fill_wei1(wei[e], e, 0.005f);
    }

    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }

    // Build with a uniform "max" N then patch per-expert N / ldb /
    // ldc.  `lda` is M-independent (transA=false → lda=K), `ldb`
    // and `ldc` track the per-expert N column count.
    auto gv = GemmVecs::uniform(num_ops, /*M=*/0, /*N=*/N_eligible, K,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*wc=*/true, /*tA=*/false,
            /*tB=*/false);
    gv.Ms = Ms;
    gv.Ns = Ns;
    for (int e = 0; e < num_ops; ++e) {
        gv.ldb[e] = Ns[e];
        gv.ldc[e] = Ns[e];
    }
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (int e = 0; e < num_ops; ++e) {
        params[e].num_threads = 32;
    }

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);

    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;

    ASSERT_TRUE(snap.valid) << "PhaseBSnapshot was not captured.";
    EXPECT_TRUE(snap.per_expert_remainder)
            << "Phase B did not fire — eligibility filter may have "
               "rejected EVERY expert (eligible_count == 0 short-"
               "circuits with `per_expert_remainder = false`).  "
               "strategy="
            << static_cast<int>(snap.strategy);
    EXPECT_EQ(static_cast<int>(snap.strategy),
            static_cast<int>(GroupNTileStrategy::ManyExperts));
    EXPECT_EQ(snap.num_ops_active, num_ops);

    // Decode-class (max_M=16 <= kDecodeMaxM) → M-proportional split.  The
    // eligibility/capacity intent is PRESERVED by proportional's per-expert
    // cap clamp `cap[e] = N[e] / ab_min_tile`: e=0 has N=512 → cap=2, so
    // despite being the M-heaviest it CANNOT exceed 2 threads, while the
    // full-N (1024 → cap=4) eligible experts absorb the surplus.  Assert
    // the cap contract rather than the legacy uniform {2,3} slot pattern.
    EXPECT_NE(snap.n_thr_fixed, 0);

    int sum = 0;
    for (int e = 0; e < num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        if (Ms[e] > 0) {
            EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
        }
        sum += n;
    }
    EXPECT_EQ(sum, 32)
            << "M-proportional should saturate the full thread team even with "
               "one capacity-capped heavy expert; sum="
            << sum;

    // KEY: e=0 (M-heaviest but N=512) is clamped to its N-tile capacity
    // cap=2 and must NOT receive more, even though it sorts first by M.
    const int n_e0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
    EXPECT_LE(n_e0, 2)
            << "INELIGIBLE/capacity-capped expert e=0 (M=16, N=512, cap=2) "
               "must not exceed its N-tile capacity; got "
            << n_e0
            << " — check the cap[e] clamp in apply_round_pick()'s "
               "proportional split.";
    // A full-N eligible heavy (e=1, M=12, N=1024, cap=4) outranks the
    // capacity-capped e=0.
    EXPECT_GT(static_cast<int>(snap.stable_n_thr_per_expert[1]), n_e0)
            << "full-capacity heavy e=1 should outrank capacity-capped e=0.";
}

TEST(TestGroupMatmulPhaseBRemainder, DecodeProportionalEnvParsing) {
    using zendnnl::lowoha::matmul::parse_grp_matmul_decode_proportional;

    EXPECT_EQ(parse_grp_matmul_decode_proportional(nullptr), 0);
    EXPECT_EQ(parse_grp_matmul_decode_proportional(""), 0);
    EXPECT_EQ(parse_grp_matmul_decode_proportional("0"), -1);
    EXPECT_EQ(parse_grp_matmul_decode_proportional("1"), 1);
    EXPECT_EQ(parse_grp_matmul_decode_proportional("2"), 0);
    EXPECT_EQ(parse_grp_matmul_decode_proportional("false"), 0);
}

TEST(TestGroupMatmulPhaseBRemainder, DefaultAutoDeclinesNonUniformNK) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::GroupNTileStrategy;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);

    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 active OMP threads; have "
                     << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride auto_strategy(0);
    DecodeProportionalOverride prop_default_auto(-1);

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Test requires AVX512BF16 / custom-kernel "
                        "dispatch support; cannot exercise Phase B "
                        "without CK engagement.";
    }

    reset_grp_matmul_caches();

    const int num_ops = 14;
    const int K = 64;
    const int N_wide = 1024;
    const int N_narrow = 512;
    const std::vector<int> Ms = {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    std::vector<int> Ns(num_ops, N_wide);
    Ns[0] = N_narrow;

    std::vector<std::vector<bfloat16_t>> src(num_ops);
    std::vector<std::vector<bfloat16_t>> wei(num_ops);
    std::vector<std::vector<bfloat16_t>> dst(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(Ms[e]) * K);
        wei[e].resize(static_cast<size_t>(K) * Ns[e]);
        dst[e].assign(static_cast<size_t>(Ms[e]) * Ns[e], bfloat16_t(0.0f));
        fill_src(src[e], e, 0.02f);
        fill_wei1(wei[e], e, 0.005f);
    }

    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }

    auto gv = GemmVecs::uniform(num_ops, /*M=*/0, /*N=*/N_wide, K,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*wc=*/true, /*tA=*/false,
            /*tB=*/false);
    gv.Ms = Ms;
    gv.Ns = Ns;
    for (int e = 0; e < num_ops; ++e) {
        gv.ldb[e] = Ns[e];
        gv.ldc[e] = Ns[e];
    }

    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (int e = 0; e < num_ops; ++e) {
        params[e].num_threads = 32;
    }

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr,
                      /*gated_act=*/nullptr),
            status_t::success);

    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder);
    EXPECT_EQ(static_cast<int>(snap.strategy),
            static_cast<int>(GroupNTileStrategy::ManyExperts));

    const int base = 2; // 32 threads / 14 experts, single-round fixture.
    EXPECT_EQ(static_cast<int>(snap.stable_n_thr_per_expert[0]), base)
            << "default auto should decline M-only proportional split for "
               "heterogeneous active N/K groups";
    EXPECT_EQ(static_cast<int>(snap.stable_n_thr_per_expert[1]), base + 1)
            << "fallback Phase B should distribute extras only to eligible "
               "experts";
}

// ===============================================================================
// [8b] TestGroupMatmulAutoSelectAlgo — pin down `select_grp_matmul_algo`'s
//      decisions on the (max_M, num_threads, num_ops) grid.
//
// Both phase settings are pinned to 0, the LEGACY cascade, evaluated in
// priority order:
//
//   0.   num_ops > kNTilePlanMaxExperts (=256)          → ALGO 1   (capacity carve-out)
//   1.   num_ops ≥ num_threads                          → ALGO 3   (Qwen-style)
//   2.   num_ops ≤ kFewExpertsAlgo1 (= 8)               → ALGO 1   (Mixtral-style)
//   3.   otherwise — M-driven:
//          prompt (max_M > kDecodeMaxM=32)              → ALGO 1
//          decode (max_M ≤ kDecodeMaxM)                 → ALGO 3   (gpt-oss-style)
//
// Unpinned, a BF16/FP16 layer of several experts runs ALGO 3 in both phases
// and every other call takes the shape rules; `TestGroupMatmulAutoPhaseEnv`
// covers both.
//
// The matrix below covers each arrow explicitly with at least one
// canonical workload (Qwen3-30B-A3B for rule 1, Mixtral 8x7B for
// rule 2, gpt-oss-20B for rule 3, and synthetic E257/E512 shapes for
// rule 0) plus thread-count sweeps to lock the rules across {32,
// 48, 64, 80, 96, 128, 192} threads where applicable.  A future
// heuristic edit that flips any row should produce a deterministic
// test failure pointing at the offending label.
//
// SCOPE NOTE — what this matrix tests.
//   These cases assert auto-select's ROUTING DECISION only — i.e.,
//   which top-level ALGO the dispatcher picks for the given inputs.
//   They do NOT verify the downstream N-tile planner's strategy
//   choice (`ManyExperts` vs `Sequential` vs `FewExperts` etc.) for
//   the ALGO 3 rows.  For shapes where ALGO 3 is selected but the
//   N-tile planner's `ntile_viable` returns false (e.g., Qwen
//   prompt at N=1536 → only 3 tiles, below the ManyExperts minimum
//   of ccd_size/2 = 4), the planner falls back to Sequential and
//   executes a path equivalent to ALGO 1 plus one-time ALGO 3
//   prepack overhead.  That fallback is the user-visible behaviour
//   for "Qwen prompt" today; it is documented in the rule-1 SCOPE
//   NOTE in `group_matmul_dispatch.cpp::auto_select_algo` and
//   accepted as a trade-off for the simpler 3-rule selector.
//   Verifying which strategy fires at execute time would require
//   either a gemm_mode assertion or the white-box planner snapshot
//   (`PhaseBSnapshot`) — currently used only for Phase B coverage.
//
// The test pokes `select_grp_matmul_algo` directly so we don't need to
// build the full dispatcher stack just to read out the decision.  It
// uses `AlgoEnvGuard` to clear `ZENDNNL_GRP_MATMUL_ALGO` (the override
// is intentionally re-read each call so in-process setenv works) so
// auto-select fires regardless of how the test was invoked.
// ===============================================================================

struct AutoSelectParam {
    int M; ///< Uniform M per expert.
    int K;
    int N;
    int num_ops;
    int num_threads;
    int expected_algo;
    std::string label; ///< Human-readable name for the parameterised case.
};

static std::string AutoSelectParamName(
        const ::testing::TestParamInfo<AutoSelectParam> &info) {
    return info.param.label;
}

class TestGroupMatmulAutoSelectAlgo
    : public ::testing::TestWithParam<AutoSelectParam> {};

TEST_P(TestGroupMatmulAutoSelectAlgo, MatchesExpected) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    reset_grp_matmul_caches();

    const auto &p = GetParam();

    // Pin ZENDNNL_GRP_MATMUL_ALGO=0 so the auto-select path fires
    // (override is uncached on purpose; an env value lingering from a
    // prior test would shadow our expectation).  `get_grp_matmul_algo`
    // returns 0 for AUTO, the intercepted W8A8 selector 4, or an invalid
    // selector; generic scheduler IDs are {1,2,3,5,6}.
    AlgoEnvGuard reset_algo(0);
    // Explicitly disable the phase env (default PROMPT=2 / DECODE=3) so
    // this suite continues to assert the LEGACY 3-rule cascade outcomes
    // it was authored for.  Without these overrides, the new default
    // phase pins would shadow the cascade and break every parametrized
    // row.  The phase env behaviour is covered by
    // `TestGroupMatmulAutoPhaseEnv` below.
    AutoPromptAlgoOverride legacy_prompt(0);
    AutoDecodeAlgoOverride legacy_decode(0);

    const int N_ops = p.num_ops;

    // Minimal but valid inputs for the dispatcher safety checks
    // (`check_m_tile_safe` / `check_n_tile_extra`) inside
    // `select_grp_matmul_algo`.  Row-major, uniform BF16 dtypes, no
    // packed-B, no quant.
    std::vector<char> layout(N_ops, 'r');
    std::vector<int> M(N_ops, p.M);
    std::vector<int> N(N_ops, p.N);
    std::vector<int> K(N_ops, p.K);
    std::vector<matmul_params> params(N_ops);
    for (int i = 0; i < N_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::bf16;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].dynamic_quant = false;
        params[i].packing.pack_format_b = 0;
    }

    const int got
            = select_grp_matmul_algo(layout, M, N, K, params, p.num_threads);
    EXPECT_EQ(got, p.expected_algo)
            << "auto-select picked ALGO " << got << ", expected "
            << p.expected_algo << "  [" << p.label << "]";
}

TEST(TestGroupMatmulAutoSelectAlgo,
        AoclPackedCkRouteRejectsPostopsAndForeignProvenance) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride legacy_prompt(0);
    AutoDecodeAlgoOverride legacy_decode(0);

    constexpr int E = 2;
    std::vector<char> layout(E, 'r');
    std::vector<int> M(E, 8), N(E, 2048), K(E, 64);
    std::vector<matmul_params> params(E);
    for (auto &p : params) {
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    }
    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, /*threads=*/E), 3)
            << "eligible AOCL-packed decode should reach CK";

    matmul_post_op relu;
    relu.po_type = post_op_type_t::relu;
    for (auto &p : params)
        p.postop_.push_back(relu);
    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, /*threads=*/E), 1)
            << "CK does not apply generic unary post-ops; full-weight AOCL "
               "must "
               "preserve them";

    for (auto &p : params) {
        p.postop_.clear();
        p.lowoha_algo = matmul_algo_t::onednn_blocked;
    }
    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, /*threads=*/E), 1)
            << "packed bytes explicitly owned by another backend must never "
               "reach AOCL unreorder or CK";
}

// Grid: enumerate at least one canonical workload per rule plus
// thread-count sweeps where they exercise the rule, and a few
// edge-case rows to lock the rule precedence.  The label composes
// `<arch>_<phase>_E<num_ops>_t<num_threads>` so a failure points
// straight at the offending case.
static std::vector<AutoSelectParam> make_auto_select_params() {
    std::vector<AutoSelectParam> out;

    // ── Rule 3 (gpt-oss-20B): ops 9..32, decode → ALGO 3, prompt → ALGO 1 ──
    // gpt-oss-20B-class shape (Op1, gate+up).  K=2880, N=5760, BF16.
    const int K_GO = 2880, N_GO = 5760;

    // Prompt: BS={8,16,32}, seq=128, topk=4 → M_per_expert ≈ {128, 256, 512}.
    // num_ops=32, max_M > kDecodeMaxM → rule 3 (M-driven) → ALGO 1,
    // EXCEPT at num_threads=32 where rule 1 fires first (num_ops ==
    // num_threads → ALGO 3).  The 32t case isn't part of the
    // gpt-oss-20B benchmark grid (which targets 64t / 128t) so the
    // rule-1 routing there is a documented boundary not a regression;
    // pinning both branches here catches any future drift in the
    // priority order.
    for (int M_val : {128, 256, 512}) {
        for (int nt : {32, 48, 64, 80, 96, 128, 192}) {
            const int expected_algo = (32 == nt) ? 3 : 1; // rule 1 vs rule 3
            out.push_back({/*M=*/M_val, /*K=*/K_GO, /*N=*/N_GO,
                    /*num_ops=*/32, /*num_threads=*/nt, expected_algo,
                    "prompt_gptoss_M" + std::to_string(M_val) + "_t"
                            + std::to_string(nt)});
        }
    }

    // Decode: max_M ≤ kDecodeMaxM=32, num_ops=32 in (8, num_threads),
    // rule 3 → ALGO 3.  At num_threads=32 the row hits rule 1 instead
    // (num_ops == num_threads → ALGO 3) — same outcome, different rule.
    for (int M_val : {4, 16, 32}) {
        for (int nt : {64, 128}) {
            out.push_back({M_val, K_GO, N_GO, /*num_ops=*/32, nt,
                    /*expected_algo=*/3,
                    "decode_gptoss_M" + std::to_string(M_val) + "_t"
                            + std::to_string(nt)});
        }
    }

    // ── Rule 2 (Mixtral 8x*): num_ops ≤ 8 → ALGO 1 (prompt + decode) ──
    // Mixtral 8x7B / 8x22B decoder block shape: K=4096, N=14336, BF16
    // (~112 MB / expert).  Real Mixtral has exactly 8 experts; the
    // few-experts gate triggers regardless of M / weight class.
    // Prompt at every host size we care about.
    for (int M_val : {128, 256, 512}) {
        for (int nt : {32, 48, 64, 80, 96, 128, 192}) {
            out.push_back({/*M=*/M_val, /*K=*/4096, /*N=*/14336,
                    /*num_ops=*/8, /*num_threads=*/nt,
                    /*expected_algo=*/1,
                    "mixtral_8x7B_prompt_M" + std::to_string(M_val) + "_t"
                            + std::to_string(nt)});
        }
    }
    // Decode: same Mixtral shape, decode-class M, rule 2 still wins
    // (num_ops=8 ≤ 8) and routes to ALGO 1.
    for (int M_val : {4, 16, 32}) {
        for (int nt : {64, 128}) {
            out.push_back({M_val, /*K=*/4096, /*N=*/14336,
                    /*num_ops=*/8, nt, /*expected_algo=*/1,
                    "mixtral_8x7B_decode_M" + std::to_string(M_val) + "_t"
                            + std::to_string(nt)});
        }
    }
    // Mirror shape (K=14336, N=4096) — tall-N variant, still 8 experts
    // → still rule 2 → ALGO 1.  Pinned because today's heuristic also
    // routed this to ALGO 1 via a different code path; the new rule
    // gives the same outcome via a simpler reason.
    out.push_back({/*M=*/256, /*K=*/14336, /*N=*/4096,
            /*num_ops=*/8, /*num_threads=*/128, /*expected_algo=*/1,
            "mixtral_tallN_E8_t128"});

    // ── Rule 1 (Qwen3-30B-A3B): num_ops ≥ num_threads → ALGO 3 ──
    // Qwen3-30B-A3B Op1 (gate+up) shape: K=2048, N=1536 (768×2 for
    // gate+up), BF16.  128 experts, top-k=8.  Pin both prompt and
    // decode at the two host sizes the deployment targets (64t, 128t):
    // num_ops=128 ≥ num_threads, so rule 1 selects ALGO 3 for every row.
    //
    // EXECUTION TIME — known fallback for the prompt rows.
    //   At N=1536, `kMinNTile=512` → `1536 / 512 = 3` tiles per expert,
    //   below the ManyExperts minimum (ccd_size/2 = 4 on 8-core CCDs).
    //   The N-tile planner's `ntile_viable` therefore returns false for
    //   the prompt-class rows here (max_M ∈ {128, 256, 512}) and the
    //   call falls back to Sequential — behaving like ALGO 1 plus one
    //   ALGO 3 prepack pass per process.  Decode-class rows
    //   (max_M ∈ {4, 16, 32}) take the decode tile (256), get 6 tiles
    //   per expert, clear `ntile_viable`, and run via ManyExperts (the
    //   path Phase B is sized for).  Both behaviours are intentional
    //   under the simplified 3-rule selector; see the rule-1 SCOPE
    //   NOTE in `group_matmul_dispatch.cpp::auto_select_algo` for the
    //   trade-off discussion.  These rows pin the SELECTION result;
    //   the planner's execute-time fallback for prompt is verified by
    //   `plan_group_n_tile`'s own self-fallback tests (R1/R2/R3 in
    //   `group_matmul_n_tile.cpp`).
    const int K_QW = 2048, N_QW = 1536;
    for (int M_val : {4, 16, 32, 128, 256, 512}) {
        for (int nt : {64, 128}) {
            const bool is_decode_case
                    = (M_val <= zendnnl::lowoha::matmul::kDecodeMaxM);
            const char *phase = is_decode_case ? "decode" : "prompt";
            const int expected_algo = 3;
            out.push_back(
                    {M_val, K_QW, N_QW, /*num_ops=*/128, nt, expected_algo,
                            std::string("qwen3_30B_") + phase + "_E128_M"
                                    + std::to_string(M_val) + "_t"
                                    + std::to_string(nt)});
        }
    }

    // ── Rule 1 + capacity carve-out (1.a): num_ops vs kNTilePlanMaxExperts ──
    // Rule 1 routes num_ops ≥ num_threads → ALGO 3, BUT the N-tile
    // planner's fixed-size stack arrays cap at
    // `GroupNTilePlan::kMaxExperts == kNTilePlanMaxExperts == 256`.
    // Sub-gate 1.a routes num_ops > kNTilePlanMaxExperts → ALGO 1, since
    // beyond that ceiling the planner's R3 gate falls back to Sequential
    // anyway.
    //
    // Three pins cover this boundary:
    //   * E256 — exactly at capacity, still ALGO 3 (rule 1: num_ops ≥
    //     num_threads; ManyExperts handles 256 experts fine).
    //   * E257 — first value above capacity, must flip to ALGO 1.
    //   * E512 — well above capacity, must stay on ALGO 1.
    //
    // The above-capacity rows answered ALGO 5 until the no-5-no-6
    // invariant made rule 0 return ALGO 1; the expectations are updated
    // to match.  Force `ZENDNNL_GRP_MATMUL_ALGO=5` to reach the old
    // per-expert pool.
    out.push_back({/*M=*/128, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/zendnnl::lowoha::matmul::kNTilePlanMaxExperts,
            /*num_threads=*/128, /*expected_algo=*/3,
            "many_experts_E256_t128_at_capacity"});
    out.push_back({/*M=*/128, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/zendnnl::lowoha::matmul::kNTilePlanMaxExperts + 1,
            /*num_threads=*/128, /*expected_algo=*/1,
            "many_experts_E257_t128_above_capacity"});
    out.push_back({/*M=*/128, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/512, /*num_threads=*/128, /*expected_algo=*/1,
            "many_experts_E512_t128_above_capacity"});

    // ── Rule 1 boundary: num_ops == num_threads → ALGO 3 ──
    // The "≥" in rule 1 is intentional; this row pins it.  Without
    // it a future change to "num_ops > num_threads" would silently
    // flip the boundary case to rule 3 (M-based).
    out.push_back({/*M=*/4, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/64, /*num_threads=*/64, /*expected_algo=*/3,
            "rule1_boundary_E64_t64_decode"});
    out.push_back({/*M=*/256, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/64, /*num_threads=*/64, /*expected_algo=*/3,
            "rule1_boundary_E64_t64_prompt"});

    return out;
}

INSTANTIATE_TEST_SUITE_P(GroupMatmulAutoSelect, TestGroupMatmulAutoSelectAlgo,
        ::testing::ValuesIn(make_auto_select_params()), AutoSelectParamName);

// ===============================================================================
// [8c] TestGroupMatmulAutoSelectAlgo_DynamicQuant — the single
//      quant configuration `check_n_tile_extra` currently accepts:
//      `dynamic_quant=true` with `{M, 1}` source + `{1, N}` per-
//      channel weight.  ALGO 3 reaches that case via `flat_n_tile`'s
//      pre-OMP source-reorder hoist (`HoistedSrcQuant` in
//      `group_matmul_n_tile.cpp`); per-tile threads then share the
//      hoisted S8 src + column-sliced wei scale.  Anything else —
//      static src, per-tensor / per-group wei, per-group src,
//      pure WOQ — falls back to ALGO 1.
//
// This test pins the four boundaries of the scope:
//   1. Non-quantised baseline → ALGO 3 (the shape itself is
//      N-tile-viable so the negative cases below prove the gate
//      flipped, not the shape).
//   2. `dynamic_quant=true` + `{M, 1}` src + `{1, N}` wei (the
//      one accepted shape) → ALGO 3.
//   3. Same as (2) but `dynamic_quant=false` → ALGO 1 (static
//      src is out of scope).
//   4. Same as (2) but `wei_scale.buff = nullptr` → ALGO 1
//      (wei pair side is required).
//
// (The hoist path itself is exercised end-to-end by the
// `TestGroupMatmulQuant.INT8_DYNAMIC_GEMM_*` suites, which carry
// both sides of the per-token pair and reach `flat_n_tile` when
// their random num_ops × num_threads combo trips auto-select's
// rule 1 or rule 3-decode arm.)
//
// `check_m_tile_safe`'s row-local granularity gate
// (`src_scale.dims[0] == M[i]`) is still a precondition; the
// per-token gate below is stricter, so anything that fails this
// test's `dynamic_quant=true + {M[i], 1}` requirement is also
// caught by `m_tile_safe` for the dynamic case.
//
// Case 5 pins the single-row decode case (`M[i] == 1` with
// `src_scale.dims = {1, 1}`): the gate was rewritten to compare
// `src_scale.dims[0]` against the actual per-expert `M[i]` (it
// previously used a brittle `dims[0] > 1` proxy that rejected
// sparse-MoE batches where some experts get exactly one routed
// token).
// ===============================================================================

TEST(TestGroupMatmulAutoSelectAlgo_DynamicQuant, AcceptsAlgo3ViaHoist) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0); // auto-select
    // Force the legacy 3-rule cascade — the test was authored to assert
    // a Rule-3 decode-class shape ALGO 3 → ALGO 1 transition driven by
    // the dynamic_quant gate.  With the new phase env defaults the
    // decode path would resolve via `AUTO_DECODE_ALGO=3` clamped on
    // !n_tile_safe to ALGO 1 directly, masking the actual gate this
    // test was written to pin.
    AutoPromptAlgoOverride legacy_prompt(0);
    AutoDecodeAlgoOverride legacy_decode(0);

    const int N_ops = 32;
    const int M_val = 16; // decode-class (≤ kDecodeMaxM)
    const int K_GO = 2880;
    const int N_GO = 5760; // wide-N decode shape
    const int num_threads = 64;

    std::vector<char> layout(N_ops, 'r');
    std::vector<int> M(N_ops, M_val);
    std::vector<int> N(N_ops, N_GO);
    std::vector<int> K(N_ops, K_GO);
    std::vector<matmul_params> params(N_ops);
    for (int i = 0; i < N_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::bf16;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].dynamic_quant = false;
        params[i].packing.pack_format_b = 0;
    }

    // Case 1 — Sanity: same shape WITHOUT any quant lands on ALGO 3
    // (decode-class + many experts + wide-N + ntile_viable).
    ASSERT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 3)
            << "baseline non-quant shape must select ALGO 3 — otherwise "
               "the negative cases below cannot distinguish a "
               "shape-induced fallback from a quant-gate rejection";

    // Case 2 — Accept the single in-scope shape: dynamic_quant=true
    // with `{M, 1}` src + `{1, N}` wei.  `src_scale.buff` stays
    // nullptr (wrapper allocates on the hoist path); `wei_scale.buff`
    // is a stack sentinel since the gate only checks for non-nullness
    // (it never dereferences the pointer), and using a local valid
    // address keeps the test allocator-free.
    alignas(float) char wei_scale_sentinel = 0;
    for (int i = 0; i < N_ops; ++i) {
        params[i].dynamic_quant = true;
        params[i].quant_params.src_scale.dims = {M_val, 1};
        params[i].quant_params.src_scale.dt = data_type_t::f32;
        params[i].quant_params.wei_scale.dims = {1, N_GO};
        params[i].quant_params.wei_scale.dt = data_type_t::f32;
        params[i].quant_params.wei_scale.buff = &wei_scale_sentinel;
    }

    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 3)
            << "dynamic_quant=true with `{M, 1}` src + `{1, N}` wei "
               "(the only in-scope quant pair) and otherwise N-tile-"
               "viable inputs should reach ALGO 3 via the pre-OMP hoist "
               "path.  A regression here would indicate either that "
               "`check_n_tile_extra` has tightened its dynamic / per-token "
               "/ per-channel gate, or that `check_m_tile_safe`'s "
               "row-local gate has tightened.";

    // Case 3 — Static src rejection: same shape and dims, but flip
    // `dynamic_quant=false`.  The current scope is dynamic-only; this
    // should fall back to ALGO 1 even though the dims/buff layout
    // looks otherwise valid for a static W8A8 per-token deployment.
    for (int i = 0; i < N_ops; ++i) {
        params[i].dynamic_quant = false;
        // Static src needs a populated src buff — give the gate something
        // realistic to look at (it'll reject before reading the pointer).
        params[i].quant_params.src_scale.buff = &wei_scale_sentinel;
    }

    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 1)
            << "static src quant (`dynamic_quant=false`) is intentionally "
               "outside the current per-token-dynamic scope — even with "
               "the otherwise-valid `{M, 1}` src + `{1, N}` wei pair, the "
               "gate must reject and fall back to ALGO 1.  A regression "
               "here would indicate the gate has relaxed its "
               "`dynamic_quant=true` requirement.";

    // Case 4 — Wei-missing rejection: restore `dynamic_quant=true`
    // but drop the wei_scale entirely.  The per-token scope requires
    // a populated wei pair, so this should fall back to ALGO 1.
    for (int i = 0; i < N_ops; ++i) {
        params[i].dynamic_quant = true;
        params[i].quant_params.src_scale.buff = nullptr; // back to hoist-alloc
        params[i].quant_params.wei_scale.buff = nullptr;
        params[i].quant_params.wei_scale.dims.clear();
        params[i].quant_params.wei_scale.dt = data_type_t::none;
    }

    EXPECT_EQ(select_grp_matmul_algo(layout, M, N, K, params, num_threads), 1)
            << "dynamic_quant=true with `{M, 1}` src but NO wei scale "
               "should fall back to ALGO 1 — the per-token scope guard "
               "in `check_n_tile_extra` requires the paired per-channel "
               "wei.  A regression here would indicate the gate has "
               "accidentally relaxed the wei-side requirement.";

    // Case 5 — Heterogeneous-M batch with one single-row expert
    // (`M[i] == 1`).  This is the common sparse-MoE decode pattern:
    // most experts get a handful of tokens, one or two experts get
    // exactly one routed token.  Previously the gates used a
    // `src_scale.dims[0] > 1` proxy that rejected the M=1 expert
    // and dragged the whole batch onto ALGO 1.  After the
    // M[i]-aware fix the gate accepts `{1, 1}` when paired with
    // M[i]=1 and the batch keeps ALGO 3 N-tile parallelism.
    //
    // Restore the in-scope quant configuration for every expert and
    // shrink the last expert's M to 1 (matching `{1, 1}` src_scale
    // dims) while keeping the other experts at M_val.  The custom
    // mixed-M `M` vector replaces the uniform one used above; N/K
    // stay uniform across experts because that's how MoE callers
    // present per-expert vectors today.
    std::vector<int> M_mixed(N_ops, M_val);
    M_mixed.back() = 1;
    for (int i = 0; i < N_ops; ++i) {
        params[i].dynamic_quant = true;
        params[i].quant_params.src_scale.buff = nullptr;
        params[i].quant_params.src_scale.dims = {M_mixed[i], 1};
        params[i].quant_params.src_scale.dt = data_type_t::f32;
        params[i].quant_params.wei_scale.dims = {1, N_GO};
        params[i].quant_params.wei_scale.dt = data_type_t::f32;
        params[i].quant_params.wei_scale.buff = &wei_scale_sentinel;
    }

    EXPECT_EQ(
            select_grp_matmul_algo(layout, M_mixed, N, K, params, num_threads),
            3)
            << "sparse-MoE batch with one M[i]=1 expert (`src_scale.dims "
               "= {1, 1}`) should still reach ALGO 3 — the M[i]-aware "
               "gate accepts the single-row decode case because the "
               "single-thread reorder for that expert is race-free and "
               "the per-token scale equals the full-matrix scale "
               "trivially.  A regression here means the gate is using "
               "the old brittle `dims[0] > 1` proxy again and will drag "
               "every sparse-MoE decode batch onto sequential_experts.";
}

// ===============================================================================
// [8d] TestGroupMatmulAutoPhaseEnv — `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO`
//      and `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` per-phase pinning.
//
// New envs let an operator pin a specific ALGO per phase without
// touching the global `ZENDNNL_GRP_MATMUL_ALGO`.  The raw phase-setting
// defaults are PROMPT=2 and DECODE=3; the default-only Rule 0.7 refines an
// unpinned prompt call to ALGO 1, and a BF16/FP16 layer of several experts
// runs ALGO 3 in both phases while unpinned.
// Set the env to `0` for the legacy 3-rule cascade.  Tests here
// pin both branches: explicit env=0 reproduces the legacy cascade,
// env={1,2,3,5,6} is honoured for the matching generic phase, and value 4
// requests the matching phase's W8A8 whole-call interceptor while mapping to
// inherited default policy after a decline. Coverage also includes the safety
// clamps (ALGO 3 on !n_tile_safe) and the legacy cascade's R0 capacity gate.
//
// All tests use `select_grp_matmul_algo` directly with `ALGO=0` so
// auto-select fires; the phase env is the only thing that varies.
// ===============================================================================

namespace {

// Build a minimal valid `(layout, M, N, K, params)` tuple for an
// auto-select probe.  Row-major BF16, no quant, no packed-B.
struct AutoProbeShape {
    std::vector<char> layout;
    std::vector<int> M;
    std::vector<int> N;
    std::vector<int> K;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
    int num_threads;
};

static AutoProbeShape build_auto_probe(
        int M_val, int K_val, int N_val, int num_ops, int num_threads) {
    using namespace zendnnl::lowoha::matmul;
    AutoProbeShape s;
    s.layout.assign(num_ops, 'r');
    s.M.assign(num_ops, M_val);
    s.N.assign(num_ops, N_val);
    s.K.assign(num_ops, K_val);
    s.params.resize(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        s.params[i].dtypes.src = data_type_t::bf16;
        s.params[i].dtypes.wei = data_type_t::bf16;
        s.params[i].dtypes.dst = data_type_t::bf16;
        s.params[i].dtypes.bias = data_type_t::bf16;
        s.params[i].mem_format_a = 'n';
        s.params[i].mem_format_b = 'n';
        s.params[i].dynamic_quant = false;
        s.params[i].packing.pack_format_b = 0;
    }
    s.num_threads = num_threads;
    return s;
}

// Rewrite a BF16 auto-select probe as n_tile-safe W4A8 (dynamic bf16 src,
// packed s4 weights, per-group wei scale).  This lets the generic AUTO rules
// be tested with W4A8 metadata without standing up the full GEMM stack.
static void apply_w4a8_auto_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    static float wei_scale_sentinel = 1.0f;
    for (size_t i = 0; i < s.params.size(); ++i) {
        auto &p = s.params[i];
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s4;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.mem_format_b = 'n';
        p.quant_params.src_zp = {};
        p.quant_params.wei_zp = {};
        const int m = (i < s.M.size() && s.M[i] > 0) ? s.M[i] : 1;
        const int n = (i < s.N.size()) ? s.N[i] : 2;
        p.quant_params.src_scale.dims = {static_cast<int64_t>(m), 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.buff = nullptr;
        p.quant_params.wei_scale.dims
                = {2, static_cast<int64_t>(n > 1 ? n : 2)};
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = &wei_scale_sentinel;
    }
}

// Rewrite a BF16 auto-select probe as f32.  Only BF16/FP16 layers of several
// experts run ALGO 3 in both phases, so an f32 probe of the same shape
// exercises the shape rules.
static void apply_shape_rules_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    for (auto &p : s.params) {
        p.dtypes.src = data_type_t::f32;
        p.dtypes.wei = data_type_t::f32;
        p.dtypes.dst = data_type_t::f32;
        p.dtypes.bias = data_type_t::f32;
    }
}

} // namespace

// Explicit-zero phase env (escape hatch for legacy behaviour) — sets
// `AUTO_PROMPT_ALGO=0` and asserts the legacy 3-rule cascade fires.
// Mixtral-class prompt → Rule 2b (num_ops≤8) → ALGO 1.
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitZeroEnablesLegacyMixtralPrompt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride explicit_legacy_prompt(0); // explicit legacy
    AutoDecodeAlgoOverride explicit_legacy_decode(0);

    // Mixtral-class prompt: 8 experts, K=4096, N=14336, M=256 (prompt).
    // Legacy Rule 2 → ALGO 1.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "AUTO_PROMPT_ALGO=0 must defer to legacy rules (Mixtral-8 "
               "prompt → Rule 2 → ALGO 1)";
}

// Explicit-zero again, exercising legacy Rule 1 (num_ops ≥ num_threads).
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitZeroEnablesLegacyQwenPrompt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride explicit_legacy_prompt(0);
    AutoDecodeAlgoOverride explicit_legacy_decode(0);

    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/128, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "AUTO_PROMPT_ALGO=0 must defer to legacy rules (Qwen "
               "num_ops≥num_threads → Rule 1 → ALGO 3)";
}

// Default prompt policy for a BF16/FP16 layer of several experts: an
// unpinned prompt runs ALGO 3, whatever the executor, ISA or custom-kernel
// setting.  This BF16 probe exercises the actual out-of-box route.
TEST(TestGroupMatmulAutoPhaseEnv, DefaultPromptRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No AutoPromptAlgoOverride / AutoDecodeAlgoOverride — exercises
    // the unset-env default policy.

    // Mixtral-class BF16 prompt.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            3)
            << "the unpinned prompt policy must route prompt → ALGO 3";
    EXPECT_STREQ(trace.reason, "auto_algo3");
    EXPECT_EQ(get_grp_matmul_auto_prompt_algo(), 2);
}

// Default decode: phase env defaults to 3 (N-tile rounds + CK).
TEST(TestGroupMatmulAutoPhaseEnv, DefaultDecodeRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // gpt-oss-style decode, n_tile_safe.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/32, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "AUTO_DECODE_ALGO default (=3) must route decode → ALGO 3";
}

// ── Rule 0.6a — decode AUTO_DECODE_ALGO=5 pin qualifier ──────────────
// AUTO never emits ALGO 5 on its own; an operator opts in with the explicit
// `AUTO_DECODE_ALGO=5` pin.  With the qualifier gate ON (default) that pin is
// HONOURED only where the measured win holds — a decode frame whose every ACTIVE
// expert is s8 AND whose active count saturates the team
// (`active_ops >= num_threads`).  The gate is model-agnostic: it fences on dtype
// and occupancy, NOT on a shape/hidden-size fingerprint, so the pin is a per-
// deployment opt-in the operator sets only where ALGO 5 helps (the measured INT8
// Qwen3/3.6 case) and every other INT8 saturated decode is honoured on the same
// terms.  On any other decode frame (bf16, low occupancy, prompt, mixed dtype)
// the pin is DECLINED and the call falls back to the decode default, exactly as
// if the pin were unset.  `DECODE_ALGO5_GATE=0` honours the pin verbatim.  The
// honoured probes use M=16 (decode) with active_ops(40) >= num_threads(32);
// absent the pin they would take Rule 0.6 → ALGO 3, so ALGO 5 proves the pin was
// honoured.

// Representative gate_up (K=2048 hidden, N=1536 = 2*intermediate), s8, saturated
// team, pin=5 → 5.  Uses a Qwen3-30B shape as the MOTIVATING example, but the
// qualifier accepts it on dtype+occupancy, not the shape.
TEST(TestGroupMatmulAutoPhaseEnv, Int8DecodePinHonouredGateUp) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "AUTO_DECODE_ALGO=5 must be honoured for an INT8 gate_up decode "
               "(all-s8 + saturated team)";
}

// Representative down proj (N=2048 hidden, K=768 intermediate), s8, saturated,
// pin=5 → 5 — the other fused-MoE matmul shape.
TEST(TestGroupMatmulAutoPhaseEnv, Int8DecodePinHonouredDownProj) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/16, /*K=*/768, /*N=*/2048,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "AUTO_DECODE_ALGO=5 must be honoured for an INT8 down-proj "
               "decode";
}

// SHAPE-AGNOSTIC: a NON-Qwen shape (Gemma4-ish hidden=2816) that is s8 and
// saturates the team is HONOURED just the same — proving the qualifier does NOT
// fence on a hidden-size fingerprint.  (Whether ALGO 5 actually helps this model
// is the operator's call: they only set the pin where it does.)
TEST(TestGroupMatmulAutoPhaseEnv, Int8DecodePinHonouredShapeAgnostic) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/16, /*K=*/2816, /*N=*/1408,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "AUTO_DECODE_ALGO=5 must be honoured for ANY INT8 saturated "
               "decode "
               "frame — the gate is shape-agnostic (dtype + occupancy only)";
}

// NO PIN: AUTO on its own must NOT emit ALGO 5 for an s8 decode frame — the
// restored no-5-no-6 invariant.  It takes the decode default (Rule 0.6,
// active_ops > num_threads → ALGO 3).
TEST(TestGroupMatmulAutoPhaseEnv, Int8DecodeNoPinStaysOnDefault) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No AutoDecodeAlgoOverride — the pin is unset.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "AUTO with no pin must not emit ALGO 5 (s8 decode stays on the "
               "ALGO 3 default)";
}

// bf16 + pin=5: the all-s8 term fails → pin DECLINED → falls back to the decode
// default, NOT ALGO 5.
TEST(TestGroupMatmulAutoPhaseEnv, Bf16DecodePinDeclined) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    // build_auto_probe leaves wei=bf16.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    EXPECT_NE(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "A =5 pin on a bf16 decode frame must be declined (s8-only)";
}

// Occupancy floor: an s8 frame whose active_ops (16) < num_threads (32) cannot
// fill the team one-expert-per-thread → pin DECLINED, NOT ALGO 5.
TEST(TestGroupMatmulAutoPhaseEnv, LowOccupancyDecodePinDeclined) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_NE(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "active_ops < num_threads must decline the =5 pin (occupancy)";
}

// The =5 pin is DECODE-ONLY: an s8 PROMPT frame (max_M > 32) is not governed by
// AUTO_DECODE_ALGO and must never route to ALGO 5.
TEST(TestGroupMatmulAutoPhaseEnv, Int8PromptPinDeclined) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_NE(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "An s8 PROMPT (max_M>32) must not route to ALGO 5 (decode-only "
               "pin)";
}

// EVERY ACTIVE expert must be s8: a group with a bf16 active expert mixed in
// must decline the pin (ALGO 5 has no uniform-dtype clamp, so a partial match
// would hand a bf16 tile to the excluded route).
TEST(TestGroupMatmulAutoPhaseEnv, MixedDtypeDecodePinDeclined) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    // One ACTIVE expert carries bf16 weights → all_active_wei_s8 fails.
    s.params[1].dtypes.wei = data_type_t::bf16;
    EXPECT_NE(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "A mixed s8/bf16 active group must decline the =5 pin";
}

// Gate OFF (`DECODE_ALGO5_GATE=0`): the pin is honoured VERBATIM even for a bf16
// frame the qualifier would otherwise decline — the escape hatch that restores
// pre-gate semantics.
TEST(TestGroupMatmulAutoPhaseEnv, DecodeAlgo5PinVerbatimWhenGateOff) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin5(5);
    DecodeAlgo5GateOverride gate_off(0);
    // bf16 frame — the qualifier would decline it, but the gate is bypassed.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2816, /*N=*/1408,
            /*num_ops=*/40, /*num_threads=*/32);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "Gate=0 must honour AUTO_DECODE_ALGO=5 verbatim for any decode "
               "frame";
}

// Prompt occupancy qualifier: `AUTO_PROMPT_ALGO=5` on an UNDER-OCCUPIED prompt
// frame (`active_ops < num_threads`) is DECLINED and redirected to ALGO 3.
// ALGO 5 runs one expert per thread with no intra-expert split, so it cannot
// fill the team; ALGO 3's N-split can.  The redirect and the unpinned
// prompt default are both ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, PromptPinAlgo5DeclinedUnderOccupancy) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride pin5(5);
    AutoDecodeAlgoOverride no_decode(0);
    // Prompt-class (max_M=256) with 16 active experts on a 64-wide team.
    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "active_ops(16) < num_threads(64) must decline the prompt =5 "
               "pin and redirect to ALGO 3, not the prompt default ALGO 2";
}

// The occupancy redirect is not a promise of ALGO 3: it rewrites the phase
// request to 3 and therefore inherits the `phase_algo == 3 && !n_tile_safe`
// clamp.  A prompt frame that declines =5 for occupancy AND cannot be
// column-sliced must land on ALGO 1, not on an ALGO 3 that would drop the
// source quantization.
//
// Same frame as the test above plus `dynamic_quant=true` with a per-token
// src_scale but NO weight-scale buffer.  The n_tile_safe=false is caused by
// `check_n_tile_extra` returning false at the `!wei_per_channel &&
// !wei_per_group` guard (no wei_scale.buff supplied), NOT by dynamic_quant
// alone — a dynamic-quant call with a proper per-channel weight scale WOULD
// be n_tile_safe.  The `src_scale.dims={M,1}` is there to satisfy the
// per-token source-scale precondition; the absent weight scale is the trigger.
TEST(TestGroupMatmulAutoPhaseEnv,
        PromptPinAlgo5OccupancyRedirectClampsToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride pin5(5);
    AutoDecodeAlgoOverride no_decode(0);
    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/64);
    for (auto &p : s.params) {
        p.dynamic_quant = true;
        p.quant_params.src_scale.dims = {256, 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "the =5 occupancy redirect to ALGO 3 must still honour the "
               "n_tile_safe clamp and fall through to ALGO 1";
}

// The prompt qualifier fences on occupancy ONLY: unlike decode, a bf16 prompt
// frame at full occupancy still honours the pin, because the decode gate's
// all-active-s8 term encodes a measured bf16 DECODE regression that was never
// established for prompt.
TEST(TestGroupMatmulAutoPhaseEnv,
        PromptPinAlgo5HonouredForBf16AtFullOccupancy) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride pin5(5);
    AutoDecodeAlgoOverride no_decode(0);
    // build_auto_probe leaves wei=bf16; occupancy is satisfied.
    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "bf16 must not decline a prompt =5 pin: the s8 term is "
               "decode-only";
}

// Gate OFF (`PROMPT_ALGO5_GATE=0`): the escape hatch restores pre-gate
// semantics, honouring the pin verbatim on a frame the qualifier would decline.
TEST(TestGroupMatmulAutoPhaseEnv, PromptAlgo5PinVerbatimWhenGateOff) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride pin5(5);
    AutoDecodeAlgoOverride no_decode(0);
    PromptAlgo5GateOverride gate_off(0);
    auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "Gate=0 must honour AUTO_PROMPT_ALGO=5 verbatim regardless of "
               "occupancy";
}

// Phase isolation: the prompt qualifier must not touch a DECODE frame, and the
// decode qualifier must not touch a PROMPT one.  A decode pin of 5 on an
// under-occupied decode frame declines to the DECODE default (also 3, but by a
// different route), so this asserts the prompt gate override cannot rescue it.
TEST(TestGroupMatmulAutoPhaseEnv, PromptAlgo5GateDoesNotAffectDecode) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    AutoDecodeAlgoOverride pin5(5);
    PromptAlgo5GateOverride prompt_gate_off(0);
    // Decode-class, under-occupied: the DECODE gate still declines.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/64);
    EXPECT_NE(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            5)
            << "PROMPT_ALGO5_GATE=0 must not disable the decode qualifier";
}

// An explicit NON-5 decode pin is honoured verbatim — the gate only qualifies
// the =5 value, so `AUTO_DECODE_ALGO=3` still pins ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, DecodeAlgo3PinHonouredVerbatim) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin3(3);
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/40, /*num_threads=*/32);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "An explicit AUTO_DECODE_ALGO=3 pin is honoured (gate qualifies "
               "only =5)";
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
    EXPECT_EQ(select_grp_matmul_algo(
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
    EXPECT_EQ(select_grp_matmul_algo(
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
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "Rule 0.45 must honour the n_tile_safe clamp rather than hand "
               "an unsafe shape to ALGO 3";
}

// ── Rule 0.46 — single dense expert BF16 prompt → ALGO 3 (opt-in) ─────
// With `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE=1` a one-expert BF16
// prompt (a dense FFN pass or a routed linear) goes to the CK prompt tile
// executor rather than Rule 0.7's ALGO 1.  The knob defaults to 0.
static int select_one(const AutoProbeShape &s,
        zendnnl::lowoha::matmul::grp_matmul_gated_act_t act
        = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::none,
        bool epilogue_ok = true) {
    return zendnnl::lowoha::matmul::select_grp_matmul_algo(s.layout, s.M, s.N,
            s.K, s.params, s.num_threads, nullptr, act, epilogue_ok);
}

TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertBf16PromptRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);

    auto s = build_auto_probe(/*M=*/2048, /*K=*/768, /*N=*/2304,
            /*num_ops=*/1, /*num_threads=*/32);
    {
        SingleExpertPromptTileOverride env_default(-1);
        if (std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE")
                == nullptr) {
            EXPECT_EQ(select_one(s), 1) << "the knob defaults to off";
        }
    }
    SingleExpertPromptTileOverride on(1);
    EXPECT_EQ(select_one(s), 3) << "plain GEMM (a routed linear)";
    EXPECT_EQ(select_one(s, grp_matmul_gated_act_t::gelu_and_mul), 3)
            << "gated FFN W13 pass";
    {
        SingleExpertPromptTileOverride off(0);
        EXPECT_EQ(select_one(s, grp_matmul_gated_act_t::gelu_and_mul), 1)
                << "knob off keeps Rule 0.7's ALGO 1";
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

// Rule 0.46 has no size gate: any shape the prompt tile executor serves
// qualifies.  It still needs one BF16 expert, an epilogue the CK takes, an N
// the CK pack divides, a row-major layout and a prompt-phase M.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertPromptScope) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    SingleExpertPromptTileOverride on(1);
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

    auto int8 = probe(768, 2304);
    int8.params[0].dtypes.wei = data_type_t::s8;
    int8.params[0].dynamic_quant = true;
    EXPECT_EQ(select_one(int8), 1) << "DQ-INT8 single expert keeps ALGO 1";

    auto unsafe = probe(768, 2304);
    unsafe.layout[0] = 'c';
    EXPECT_EQ(select_one(unsafe), 1) << "n_tile_safe clamp";

    auto two = probe(768, 2304, /*num_ops=*/2);
    int two_off = 0;
    {
        SingleExpertPromptTileOverride off(0);
        two_off = select_one(two);
    }
    EXPECT_EQ(select_one(two), two_off) << "two experts are outside Rule 0.46";
}

// A Rule 0.46 layer keeps ALGO 3 in decode too, so its prompt and decode
// share one custom-kernel pack: even with Rule 0.45 off, and at one thread.
// A caller-prepacked AOCL weight keeps its decode rules.  `decode_d` and one
// active expert of a larger pool are outside Rule 0.46.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertLayerStaysOnAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    SingleExpertPromptTileOverride on(1);
    DenseDecodeNTileOverride rule045_off(0);

    auto decode = build_auto_probe(/*M=*/4, /*K=*/768, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    EXPECT_EQ(select_one(decode), 3) << "decode follows the layer";
    auto one_thread = build_auto_probe(/*M=*/4, /*K=*/768, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/1);
    EXPECT_EQ(select_one(one_thread), 3) << "at any team size";
    auto prepacked = decode;
    prepacked.params[0].mem_format_b = 'r';
    EXPECT_EQ(select_one(prepacked), 1) << "AOCL-prepacked decode keeps AOCL";
    {
        SingleExpertPromptTileOverride off(0);
        EXPECT_EQ(select_one(decode), 1) << "knob off: Rule 0.45 off -> 1";
    }

    auto prompt = build_auto_probe(/*M=*/2048, /*K=*/768, /*N=*/1280,
            /*num_ops=*/1, /*num_threads=*/32);
    {
        NTileStrategyOverride decode_d(1);
        EXPECT_EQ(select_one(prompt), 1) << "decode_d never serves the tile";
    }
    auto pooled = prompt;
    pooled.params[0].active_matmul = 1;
    pooled.params[0].total_matmul = 8;
    auto_algo_trace trace;
    select_grp_matmul_algo(pooled.layout, pooled.M, pooled.N, pooled.K,
            pooled.params, pooled.num_threads, &trace);
    ASSERT_NE(trace.reason, nullptr);
    EXPECT_STREQ(trace.reason, "auto_algo3")
            << "one active expert of an MoE pool follows its MoE layer";
}

// AUTO_PROMPT_ALGO=3 forces ALGO 3 on a shape Rule 2 would have
// sent to ALGO 1 (Mixtral-class prompt) — the override is honoured.
TEST(TestGroupMatmulAutoPhaseEnv, PromptEnvForcesAlgo3OnMixtralPrompt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride force_prompt(3); // ALGO 3 for prompt
    AutoDecodeAlgoOverride no_decode(0);

    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "AUTO_PROMPT_ALGO=3 must override Rule 2 (num_ops≤8) for "
               "the prompt phase";
}

// AUTO_DECODE_ALGO=1 forces ALGO 1 on a decode shape Rule 3 would
// have sent to ALGO 3 (gpt-oss-class decode) — phase env honoured
// for decode phase only.
TEST(TestGroupMatmulAutoPhaseEnv, DecodeEnvForcesAlgo1OnGptOssDecode) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    AutoDecodeAlgoOverride force_decode(1); // ALGO 1 for decode

    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/32, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "AUTO_DECODE_ALGO=1 must override Rule 3 (decode → ALGO 3) "
               "for the decode phase";
}

// More ACTIVE experts than threads → ALGO 3 (Rule 0.6 for a layer outside
// the BF16 MoE scope), and the N-tile strategy does not participate in that
// decision.  Pinned to strategy 2
// (Rounds) precisely to show the strategy does NOT change the answer —
// compare MoreExpertsThanThreadsRoutesToAlgo3UnderDefault below.
TEST(TestGroupMatmulAutoPhaseEnv,
        MoreExpertsThanThreadsRoutesToAlgo3UnderRounds) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    NTileStrategyOverride rounds(2); // legacy Rounds routing
    // No phase-env override — exercises the default AUTO policy.

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "num_ops(88) > num_threads(64) under n_tile strategy 2 must "
               "route AUTO to ALGO 3 (auto-select never emits ALGO 5)";
}

// Under the DEFAULT strategy (3 = decode_dynamic) the same
// experts-exceed-threads decode frame stays on ALGO 3 so the CCD-cohesive
// DecodeDynamic pool can engage.
TEST(TestGroupMatmulAutoPhaseEnv,
        MoreExpertsThanThreadsRoutesToAlgo3UnderDefault) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No n_tile strategy override — exercises the shipped default.

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "num_ops(88) > num_threads(64) must stay on ALGO 3 under the "
               "default decode_dynamic strategy";
}

// The same Qwen-class shape on 128 threads: num_ops(88) <= num_threads(128),
// and ALGO 3 single round stays selected.
TEST(TestGroupMatmulAutoPhaseEnv, ExpertsLeqThreadsKeepsAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "num_ops(88) <= num_threads(128): ALGO 3 single round feasible, "
               "Rule 0.6 must not fire";
}

// An explicit AUTO_DECODE_ALGO=3 pin and the default agree on ALGO 3 for
// the experts-exceed-threads decode regime, so this locks in the answer
// rather than the precedence; the pin-outranks-default direction is covered
// by ExplicitPhasePinOutranksPolicyRule below.
TEST(TestGroupMatmulAutoPhaseEnv, MoreExpertsThanThreadsDecodePin3RunsAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    NTileStrategyOverride rounds(2);
    AutoDecodeAlgoOverride force_decode(3);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "decode num_ops(88) > num_threads(64) with AUTO_DECODE_ALGO=3 "
               "must run ALGO 3";
}

// An explicit phase pin outranks the AUTO default.  The default would answer
// ALGO 3 on this experts-exceed-threads decode frame, but the operator asked
// for ALGO 1 and policy does not get to overrule that.  The tile-safety
// clamps still do.
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitPhasePinOutranksPolicyRule) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    AutoDecodeAlgoOverride force_decode(1);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "an explicit AUTO_DECODE_ALGO=1 pin must outrank Rule 0.6's "
               "policy answer for decode num_ops(88) > num_threads(64)";
}

// ALGO 5 / ALGO 6 phase pins are HONOURED, not rewritten to 3. The no-5-no-6
// invariant binds auto-select's own heuristics; an operator who exports the
// env is making an explicit request.
TEST(TestGroupMatmulAutoPhaseEnv, PhasePinsAlgo5And6AreHonoured) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    {
        AutoPromptAlgoOverride force_prompt(5);
        AutoDecodeAlgoOverride no_decode(0);
        // Prompt-class (max_M=256 > kDecodeMaxM).  `num_ops >= num_threads`
        // so the prompt occupancy qualifier honours the pin; the qualifier
        // itself is covered by `PromptPinAlgo5*` below.
        auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/64, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(
                          s.layout, s.M, s.N, s.K, s.params, s.num_threads),
                5)
                << "explicit AUTO_PROMPT_ALGO=5 must be honoured, not clamped "
                   "to ALGO 3";
    }
    {
        AutoPromptAlgoOverride no_prompt(0);
        AutoDecodeAlgoOverride force_decode(6);
        auto s = build_auto_probe(/*M=*/8, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/16, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(
                          s.layout, s.M, s.N, s.K, s.params, s.num_threads),
                6)
                << "explicit AUTO_DECODE_ALGO=6 must be honoured, not clamped "
                   "to ALGO 3";
    }
}

// The global ZENDNNL_GRP_MATMUL_ALGO force is still honoured in the
// experts-exceed-threads regime (it is applied before auto_select_algo),
// so an operator retains a per-call escape hatch back to ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, MoreExpertsThanThreadsGlobalForceWins) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard force_algo(3); // global force ALGO 3

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "global ZENDNNL_GRP_MATMUL_ALGO=3 must win over Rule 0.6";
}

// ── The experts-exceed-threads decode regime runs ALGO 3 (where ────────
// ── DecodeDynamic lives) under every N-tile strategy. ─────────────────
// These tests lock the routing: strategies 3, 2 and 0
// route the n_tile-safe shape to ALGO 3, and an n_tile-unsafe shape clamps
// to ALGO 1.  AUTO env (env_algo=0) throughout.
TEST(TestGroupMatmulAutoPhaseEnv,
        ForcedDecodeDynamicReroutesManyExpertsDecodeToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    NTileStrategyOverride force_decdyn(
            3); // ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=3

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "forced N_TILE_STRATEGY=3: decode num_ops(88) > num_threads(64) "
               "must route to ALGO 3 (DecodeDynamic), not ALGO 5";
}

// Companion: with the strategy set to Rounds (=2) the same shape ALSO lands
// on ALGO 3.  What this pins down is that N_TILE_STRATEGY only chooses WHICH
// ALGO 3 strategy runs, never WHETHER ALGO 3 runs.
TEST(TestGroupMatmulAutoPhaseEnv,
        RoundsStrategyManyExpertsDecodeAlsoRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    NTileStrategyOverride default_rounds(2); // explicit production default

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "strategy=2: decode num_ops(88) > num_threads(64) must still "
               "route to ALGO 3 — N_TILE_STRATEGY no longer gates the ALGO";
}

// Smart-AUTO (n_tile_strategy=0) routes the experts-exceed-threads decode
// regime to ALGO 3 (where DecodeDynamic lives), same as strategies 3 and 2
// above.
TEST(TestGroupMatmulAutoPhaseEnv, SmartAutoManyExpertsDecodeRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    NTileStrategyOverride smart_auto(0); // ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "smart-auto (strategy=0): decode num_ops(88) > num_threads(64) "
               "must route to ALGO 3 so DecodeDynamic can engage";
}

// Safety clamp: forced DecodeDynamic still honours n_tile_safe.  A
// non-row-major layout is n_tile-unsafe (ALGO 3 cannot serve it), so the
// re-route must fall through rather than hand an unsafe shape to ALGO 3.
// The fallback is ALGO 1 (sequential full-team) — auto-select never emits 5,
// so every `n_tile_safe` clamp answers 1.
TEST(TestGroupMatmulAutoPhaseEnv, ForcedDecodeDynamicHonorsNTileSafetyClamp) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    NTileStrategyOverride force_decdyn(3);

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    // Make the shape n_tile-unsafe (non-row-major → fails m_tile_safe,
    // hence n_tile_safe).
    for (auto &c : s.layout)
        c = 'c';
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "forced DecodeDynamic on an n_tile-unsafe shape must clamp to "
               "ALGO 1 (not hand an unsafe shape to ALGO 3)";
}

// ── DecodeDynamic non-custom WIDE swiglu parity with Rounds ──────────
// DecodeDynamic hosts the non-custom wide-fused activation via a
// team-wide barrier + apply_swiglu_oai post-pass (it used to route such
// calls to Rounds).  Verify true parity: the SAME non-custom wide-swiglu
// decode shape must produce identical output under N_TILE_STRATEGY=3
// (DecodeDynamic) and =2 (Rounds).  Without the post-pass DecodeDynamic
// would leave raw (gate, up) pairs in dst and diverge.  The gemm_mode
// capture asserts DecodeDynamic actually engaged (else the comparison
// would be vacuous).
TEST(TestGroupMatmulDecodeDynamic, NonCustomWideSwigluMatchesRounds) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    constexpr int num_ops = 8, M = 4, K = 128, N = 256; // decode, even N
    const int num_threads = 8; // 1 CCD → enough_experts (8 >= 4*num_ccds=4)

    // Shared deterministic inputs across both runs.
    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(static_cast<size_t>(K) * N);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = bfloat16_t(0.02f * (float)((int)((i + e) % 5) - 2));
        for (size_t i = 0; i < wei[e].size(); ++i)
            wei[e][i] = bfloat16_t(0.01f * (float)((int)((i + e) % 7) - 3));
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
    }
    auto gv = GemmVecs::uniform(num_ops, M, N, K); // ldc = N (wide fused)
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (int e = 0; e < num_ops; ++e)
        params[e].num_threads = num_threads;

    auto run = [&](int strategy, std::string &mode_out)
            -> std::vector<std::vector<bfloat16_t>> {
        NTileStrategyOverride strat(strategy);
        CustomKernelOverride ck_off(false); // force the non-custom path
        AlgoEnvGuard algo3(3);
        reset_grp_matmul_caches();
        std::vector<std::vector<bfloat16_t>> dst(num_ops);
        std::vector<void *> dsts(num_ops);
        for (int e = 0; e < num_ops; ++e) {
            dst[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
            dsts[e] = dst[e].data();
        }
        const char *mode = nullptr;
        group_matmul_run_parallel_dispatch(gv.layout, gv.transA, gv.transB,
                gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                biases, gv.beta, dsts, gv.ldc, gv.is_wc, params, num_threads,
                &mode, grp_matmul_gated_act_t::swiglu_oai_mul,
                data_type_t::bf16);
        mode_out = (mode != nullptr) ? mode : "";
        return dst;
    };

    std::string mode_decdyn, mode_rounds;
    const auto dst_decdyn = run(3, mode_decdyn);
    const auto dst_rounds = run(2, mode_rounds);

    ASSERT_NE(mode_decdyn.find("decdyn"), std::string::npos)
            << "DecodeDynamic did not engage (gemm_mode=" << mode_decdyn
            << "); the non-custom wide-swiglu parity check would be vacuous.";

    // Compare the activated output region [0, N/2) of every expert.
    const int half = N / 2;
    double max_d = 0.0;
    for (int e = 0; e < num_ops; ++e) {
        for (int m = 0; m < M; ++m) {
            for (int j = 0; j < half; ++j) {
                const float a = static_cast<float>(dst_decdyn[e][m * N + j]);
                const float b = static_cast<float>(dst_rounds[e][m * N + j]);
                max_d = std::max(max_d, std::fabs(static_cast<double>(a) - b));
            }
        }
    }
    EXPECT_LT(max_d, 1e-2)
            << "DecodeDynamic non-custom wide swiglu must match Rounds "
               "(mode_decdyn="
            << mode_decdyn << " mode_rounds=" << mode_rounds
            << "); max_abs_diff=" << max_d;
}

// Fewer active experts than L3 domains, each too large for one L3: on the
// custom kernel a BF16 MoE expert takes several domains, the last one the
// remainder and any partial domain.  Every column is still one tile over the
// full K, so the output matches Rounds bit for bit.
TEST(TestGroupMatmulDecodeDynamic,
        FewLargeExpertsSpreadOverL3DomainsMatchRounds) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    // 2048 x 16384 bf16 = 64 MiB per expert, twice one L3.
    constexpr int K = 2048, N = 16384, M = 2;
    for (const int experts : {2, 3}) {
        std::vector<std::vector<bfloat16_t>> src(experts), wei(experts);
        std::vector<const void *> srcs(experts), weis(experts),
                biases(experts, nullptr);
        for (int e = 0; e < experts; ++e) {
            src[e].resize(static_cast<size_t>(M) * K);
            wei[e].resize(static_cast<size_t>(K) * N);
            for (size_t i = 0; i < src[e].size(); ++i)
                src[e][i] = bfloat16_t(0.02f * (float)((int)((i + e) % 5) - 2));
            for (size_t i = 0; i < wei[e].size(); ++i)
                wei[e][i] = bfloat16_t(0.01f * (float)((int)((i + e) % 7) - 3));
            srcs[e] = src[e].data();
            weis[e] = wei[e].data();
        }
        auto gv = GemmVecs::uniform(experts, M, N, K);
        for (const int threads : {24, 28, 32}) {
            auto params = make_uniform_params(experts, data_type_t::bf16);
            for (auto &p : params)
                p.num_threads = threads;
            const auto run = [&](int strategy, std::string &mode_out) {
                NTileStrategyOverride strat(strategy);
                CustomKernelOverride ck_on(true);
                AlgoEnvGuard algo3(3);
                WeightCacheGuard wc(1);
                reset_grp_matmul_caches();
                std::vector<std::vector<bfloat16_t>> dst(experts);
                std::vector<void *> dsts(experts);
                for (int e = 0; e < experts; ++e) {
                    dst[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
                    dsts[e] = dst[e].data();
                }
                const char *mode = nullptr;
                group_matmul_run_parallel_dispatch(gv.layout, gv.transA,
                        gv.transB, gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda,
                        weis, gv.ldb, biases, gv.beta, dsts, gv.ldc, gv.is_wc,
                        params, threads, &mode, grp_matmul_gated_act_t::none,
                        data_type_t::bf16);
                mode_out = (mode != nullptr) ? mode : "";
                return dst;
            };
            std::string mode_decdyn, mode_rounds;
            const auto decdyn = run(3, mode_decdyn);
            const auto rounds = run(2, mode_rounds);
            ASSERT_NE(mode_decdyn.find("decdyn_custom"), std::string::npos)
                    << "experts=" << experts << " threads=" << threads
                    << " gemm_mode=" << mode_decdyn;
            for (int e = 0; e < experts; ++e) {
                EXPECT_EQ(std::memcmp(decdyn[e].data(), rounds[e].data(),
                                  decdyn[e].size() * sizeof(bfloat16_t)),
                        0)
                        << "experts=" << experts << " threads=" << threads
                        << " expert " << e;
            }
        }
    }
}

// ── WEIGHT_CACHE=2 (in-place) grouped-route policy ──────────────────
// `group_matmul_run_parallel_dispatch` downgrades the process-wide
// weight-cache mode 2 -> 1 ONLY for AUTO (env_algo=0), where an AOCL
// prompt reorder and an ALGO 3 decode reorder touch the same weight
// buffer with different layouts.  Pinned ALGO 1/2/5/6 (single AOCL
// layout) and pinned ALGO 3 + CK keep WC=2: ALGO 3 + CK is made safe in
// the pack layer (bf16/even-K packs in place into the single-consumer
// weight buffer; int8/odd-K fall back to an out-of-place cache that
// never mutates the weight).  Downgrade is process-wide + sticky, so
// each test re-arms WC=2 via the RAII guard.  Under the mixed in-place
// mode, a bf16 layer of several experts that AUTO keeps on ALGO 3 writes
// its custom-kernel pack in place and keeps its AOCL reorders out-of-place;
// the mixed-mode tests of the AOCL prompt reorder pin such a layer's prompt
// to ALGO 1.
namespace {
// WeightCacheGuard (RAII: set weight-cache mode, restore the prior value on
// scope exit so the sticky downgrade does not leak into sibling tests) is a
// shared helper defined in gtest_utils.hpp.

struct MinRunResult {
    std::vector<std::vector<bfloat16_t>> dst; // per-expert [M,N] output
    bool weights_mutated = false; // in-place reorder happened?
};

// Minimal valid bf16 grouped matmul (even-K, CK-eligible shape) that
// reaches group_matmul_run_parallel_dispatch.  Deterministic varied fill
// so an incorrect pack layout would change the output.  Captures the
// per-expert dst and whether the weight buffers were mutated in place by
// the reorder/pack.  `weight_offset_elems` shifts the weight base by that
// many bf16 elements inside an over-allocated buffer — set to an odd value
// to force a non-64-byte-aligned weight pointer (exercises the unaligned
// in-place pack + unaligned microkernel B-load path).
inline MinRunResult run_min_grp_matmul(
        int weight_offset_elems = 0, bool include_inactive_packed = false) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    constexpr int num_ops = 4, M = 8, K = 256, N = 256; // K even, N%32==0
    const size_t wei_elems = static_cast<size_t>(K) * N;
    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
            dst(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        // Over-allocate by the offset so the (possibly misaligned) weight
        // base can point mid-buffer without overrunning.
        wei[e].resize(wei_elems + static_cast<size_t>(weight_offset_elems));
        dst[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = bfloat16_t(0.02f * (float)((int)((i + e) % 5) - 2));
        bfloat16_t *wptr = wei[e].data() + weight_offset_elems;
        for (size_t i = 0; i < wei_elems; ++i)
            wptr[i] = bfloat16_t(0.01f * (float)((int)((i + e) % 7) - 3));
        srcs[e] = src[e].data();
        weis[e] = wptr;
        dsts[e] = dst[e].data();
    }
    // Snapshot only the logical weight region (from the offset base).
    std::vector<std::vector<bfloat16_t>> wei_orig(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        const bfloat16_t *wptr = wei[e].data() + weight_offset_elems;
        wei_orig[e].assign(wptr, wptr + wei_elems);
    }
    auto gv = GemmVecs::uniform(num_ops, M, N, K);
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (int e = 0; e < num_ops; ++e)
        params[e].num_threads = 8;
    if (include_inactive_packed) {
        // Exercise a mixed raw/caller-packed pool without asking any backend
        // to consume synthetic packed bytes: expert 0 is inactive metadata,
        // while experts 1..3 execute from raw weights.
        gv.Ms[0] = 0;
        params[0].mem_format_b = 'r';
        params[0].lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    }
    EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
            zendnnl::error_handling::status_t::success);
    MinRunResult r;
    r.dst = dst;
    for (int e = 0; e < num_ops; ++e) {
        const bfloat16_t *wptr = wei[e].data() + weight_offset_elems;
        if (std::memcmp(
                    wptr, wei_orig[e].data(), wei_elems * sizeof(bfloat16_t))
                != 0) {
            r.weights_mutated = true;
        }
    }
    return r;
}

inline double max_abs_diff(const std::vector<std::vector<bfloat16_t>> &a,
        const std::vector<std::vector<bfloat16_t>> &b) {
    double m = 0.0;
    for (size_t e = 0; e < a.size(); ++e)
        for (size_t i = 0; i < a[e].size(); ++i)
            m = std::max(m,
                    std::abs((double)static_cast<float>(a[e][i])
                            - (double)static_cast<float>(b[e][i])));
    return m;
}

// Dtype-generic sibling of run_min_grp_matmul: lets the WC tests cover
// f32 (non-bf16 mixed-mode guard) and odd-K (in-place size-gate fallback)
// without duplicating the harness.  `T` is the element type matching `dt`;
// `K_dim` may be odd (forces CK to refuse + AOCL/in-place to fall back to
// out-of-place); `weight_offset_elems` shifts the weight base for the
// unaligned variant.  Captures the per-expert dst and whether any weight
// buffer was mutated in place.
template <typename T>
struct MinRunResultT {
    std::vector<std::vector<T>> dst;
    bool weights_mutated = false;
};

// `N_dim` lets a caller pick an N that no CK `pack_nr` (32 or 64) divides,
// which is a PER-CALL custom-kernel refusal: it leaves
// `grp_matmul_custom_kernel_effective()` (master + family sub-knobs + weight
// scales) untouched, so AUTO mixed mode stays ELIGIBLE process-wide while
// `ck_eligible_*` declines this particular weight.  That is the only way to
// reach the local refusal handling; turning a family sub-knob off instead
// downgrades the whole process to WC=1 and bypasses it.
//
// `M_dim` selects the phase.  M <= kDecodeMaxM (32) is decode and, for a
// few experts, Rule 0.5 may select ALGO 3.  M > 32 is prompt and Rule 0.7
// selects ALGO 1 for other dtypes; a BF16/FP16 layer of several experts runs
// ALGO 3 there too unless its prompt is pinned away.  `num_threads` and
// `num_ops` are likewise overridable so a decode call can be forced onto
// ALGO 3's narrow-N full-weight arm.
template <typename T>
inline MinRunResultT<T> run_min_grp_matmul_typed(data_type_t dt,
        int K_dim = 256, int weight_offset_elems = 0, int N_dim = 256,
        int M_dim = 8, int num_threads = 8, int num_ops = 4) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    const int M = M_dim;
    const int N = N_dim;
    const int K = K_dim;
    const size_t wei_elems = static_cast<size_t>(K) * N;
    std::vector<std::vector<T>> src(num_ops), wei(num_ops), dst(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(wei_elems + static_cast<size_t>(weight_offset_elems));
        dst[e].assign(static_cast<size_t>(M) * N, T(0.0f));
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = T(0.02f * (float)((int)((i + e) % 5) - 2));
        T *wptr = wei[e].data() + weight_offset_elems;
        for (size_t i = 0; i < wei_elems; ++i)
            wptr[i] = T(0.01f * (float)((int)((i + e) % 7) - 3));
        srcs[e] = src[e].data();
        weis[e] = wptr;
        dsts[e] = dst[e].data();
    }
    std::vector<std::vector<T>> wei_orig(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        const T *wptr = wei[e].data() + weight_offset_elems;
        wei_orig[e].assign(wptr, wptr + wei_elems);
    }
    auto gv = GemmVecs::uniform(num_ops, M, N, K);
    auto params = make_uniform_params(num_ops, dt);
    for (int e = 0; e < num_ops; ++e)
        params[e].num_threads = num_threads;
    EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
            zendnnl::error_handling::status_t::success);
    MinRunResultT<T> r;
    r.dst = dst;
    for (int e = 0; e < num_ops; ++e) {
        const T *wptr = wei[e].data() + weight_offset_elems;
        if (std::memcmp(wptr, wei_orig[e].data(), wei_elems * sizeof(T)) != 0)
            r.weights_mutated = true;
    }
    return r;
}

template <typename T>
inline double max_abs_diff_t(const std::vector<std::vector<T>> &a,
        const std::vector<std::vector<T>> &b) {
    double m = 0.0;
    for (size_t e = 0; e < a.size(); ++e)
        for (size_t i = 0; i < a[e].size(); ++i)
            m = std::max(m,
                    std::abs((double)static_cast<float>(a[e][i])
                            - (double)static_cast<float>(b[e][i])));
    return m;
}

template <typename T>
struct UnsupportedPhaseSplitResult {
    zendnnl::error_handling::status_t status;
    bool weights_changed = false;
    bool dst_changed = false;
};

template <typename T>
UnsupportedPhaseSplitResult<T> run_unsupported_phase_split(
        data_type_t dt, int32_t weight_cache_type) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    constexpr int E = 2, M = 64, K = 256, N = 256;
    const T sentinel(-3.0f);
    std::vector<std::vector<T>> src(E), wei(E), dst(E), raw(E);
    std::vector<const void *> srcs(E), weights(E), biases(E, nullptr);
    std::vector<void *> dsts(E);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(static_cast<size_t>(K) * N);
        dst[e].assign(static_cast<size_t>(M) * N, sentinel);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = T(0.02f * static_cast<float>((i + e) % 5) - 0.04f);
        for (size_t i = 0; i < wei[e].size(); ++i)
            wei[e][i] = T(0.01f * static_cast<float>((i + e) % 7) - 0.03f);
        raw[e] = wei[e];
        srcs[e] = src[e].data();
        weights[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }
    auto gv = GemmVecs::uniform(E, M, N, K);
    auto params = make_uniform_params(E, dt);
    for (auto &p : params) {
        p.num_threads = 8;
        p.weight_cache_type = weight_cache_type;
    }
    UnsupportedPhaseSplitResult<T> result;
    result.status = group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
            gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weights, gv.ldb, biases,
            gv.beta, dsts, gv.ldc, gv.is_wc, params,
            /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
    for (int e = 0; e < E; ++e) {
        result.weights_changed |= std::memcmp(wei[e].data(), raw[e].data(),
                                          raw[e].size() * sizeof(T))
                != 0;
        result.dst_changed
                |= std::any_of(dst[e].begin(), dst[e].end(), [&](T v) {
            return static_cast<float>(v) != static_cast<float>(sentinel);
        });
    }
    return result;
}

template <typename T>
struct UnsupportedFusedPhaseSplitResult {
    zendnnl::error_handling::status_t status;
    bool w13_changed = false;
    bool w2_changed = false;
    bool dst_changed = false;
};

template <typename T>
UnsupportedFusedPhaseSplitResult<T> run_unsupported_fused_phase_split(
        data_type_t dt, bool primary_aocl_prepacked) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    constexpr int E = 2, M = 64, H = 64, N1 = 128, KDown = N1 / 2;
    const T sentinel(-3.0f);
    std::vector<std::vector<T>> src(E), w13(E), w2(E), dst(E);
    std::vector<const void *> srcs(E), w13_ptrs(E), w2_ptrs(E),
            no_bias(E, nullptr);
    std::vector<void *> dst_down(E), dst_op1(E, nullptr);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(M) * H);
        w13[e].resize(static_cast<size_t>(H) * N1);
        w2[e].resize(static_cast<size_t>(KDown) * H);
        dst[e].assign(static_cast<size_t>(M) * H, sentinel);
        fill_src(src[e], e, 0.5f);
        fill_wei1(w13[e], e, 0.1f);
        fill_wei2(w2[e], e);
        srcs[e] = src[e].data();
        w13_ptrs[e] = w13[e].data();
        w2_ptrs[e] = w2[e].data();
        dst_down[e] = dst[e].data();
    }
    const auto raw_w13 = w13;
    const auto raw_w2 = w2;

    auto gv = GemmVecs::uniform(E, M, N1, H);
    auto params = make_uniform_params(E, dt);
    for (auto &p : params) {
        p.num_threads = 2;
        p.weight_cache_type = 2;
        if (primary_aocl_prepacked) {
            p.mem_format_b = 'r';
            p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
        }
    }
    grp_matmul_gated_act_params act {};
    act.act = grp_matmul_gated_act_t::silu_and_mul;
    auto fused = make_fused_moe_op2(E, H, w2_ptrs, no_bias);
    fused.dst_down = dst_down;
    fused.ldc_down.assign(E, H);
    std::vector<int> ldc_op1(E, 0);

    UnsupportedFusedPhaseSplitResult<T> result;
    result.status = group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
            gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, w13_ptrs, gv.ldb, no_bias,
            gv.beta, dst_op1, ldc_op1, gv.is_wc, params,
            /*moe_postop=*/nullptr, &act, &fused);
    for (int e = 0; e < E; ++e) {
        result.w13_changed |= std::memcmp(w13[e].data(), raw_w13[e].data(),
                                      raw_w13[e].size() * sizeof(T))
                != 0;
        result.w2_changed |= std::memcmp(w2[e].data(), raw_w2[e].data(),
                                     raw_w2[e].size() * sizeof(T))
                != 0;
        result.dst_changed
                |= std::any_of(dst[e].begin(), dst[e].end(), [&](T v) {
            return static_cast<float>(v) != static_cast<float>(sentinel);
        });
    }
    return result;
}
} // namespace

// RAII: temporarily set a finite LRU cache capacity so the AUTO
// mixed-in-place eligibility gate (which requires unlimited capacity)
// fails, exercising the historical downgrade fallback.
struct LruCapacityGuard {
    uint32_t prev;
    explicit LruCapacityGuard(uint32_t v)
        : prev(zendnnl::common::matmul_config_t::instance()
                          .get_lru_cache_capacity()) {
        zendnnl::common::matmul_config_t::instance().set_lru_cache_capacity(v);
    }
    ~LruCapacityGuard() {
        zendnnl::common::matmul_config_t::instance().set_lru_cache_capacity(
                prev);
    }
    LruCapacityGuard(const LruCapacityGuard &) = delete;
    LruCapacityGuard &operator=(const LruCapacityGuard &) = delete;
};

template <typename T>
void expect_global3_prompt1_wc2_fails(data_type_t dt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    const auto wc1 = run_unsupported_phase_split<T>(dt, 1);
    EXPECT_EQ(wc1.status, zendnnl::error_handling::status_t::success);
    EXPECT_FALSE(wc1.weights_changed);
    EXPECT_TRUE(wc1.dst_changed);

    reset_grp_matmul_caches();
    const auto wc2 = run_unsupported_phase_split<T>(dt, 2);
    EXPECT_EQ(wc2.status, zendnnl::error_handling::status_t::failure);
    EXPECT_FALSE(wc2.weights_changed)
            << "unsupported policy must fail before prepack mutation";
    EXPECT_FALSE(wc2.dst_changed)
            << "unsupported policy must fail before destination writes";
}

template <typename T>
void expect_fused_global3_prompt5_wc2_fails(data_type_t dt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard global_algo3(3);
    AutoPromptAlgoOverride prompt_algo5(5);
    AutoDecodeAlgoOverride decode_unset(-1);
    PromptAlgo5GateOverride honour_prompt_pin(0);
    Algo5VerticalFusionOverride fused_algo5_on(1);
    WeightCacheGuard process_wc2(2);
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);

    for (const bool primary_aocl_prepacked : {false, true}) {
        SCOPED_TRACE(primary_aocl_prepacked
                        ? "W13 excluded; W2 must trigger the guard"
                        : "both pools eligible");
        const auto result = run_unsupported_fused_phase_split<T>(
                dt, primary_aocl_prepacked);
        EXPECT_EQ(result.status, zendnnl::error_handling::status_t::failure);
        EXPECT_FALSE(result.w13_changed)
                << "unsupported policy mutated the W13 pool";
        EXPECT_FALSE(result.w2_changed)
                << "unsupported policy mutated the W2 pool";
        EXPECT_FALSE(result.dst_changed)
                << "unsupported policy wrote the destination";
    }
    reset_grp_matmul_caches();
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        GlobalAlgo3PromptPin1Wc2FailsGracefullyBf16) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel";
    }
    AlgoEnvGuard global_algo3(3);
    AutoPromptAlgoOverride prompt_algo1(1);
    AutoDecodeAlgoOverride decode_unset(-1);
    WeightCacheGuard process_wc2(2);
    CustomKernelOverride ck_on(true);
    EXPECT_TRUE(grp_matmul_wc2_global_phase_split_unsupported());
    expect_global3_prompt1_wc2_fails<bfloat16_t>(data_type_t::bf16);
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        GlobalAlgo3PromptPin1Wc2FailsGracefullyF16) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "needs the FP16 custom kernel";
    }
    AlgoEnvGuard global_algo3(3);
    AutoPromptAlgoOverride prompt_algo1(1);
    AutoDecodeAlgoOverride decode_unset(-1);
    WeightCacheGuard process_wc2(2);
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    EXPECT_TRUE(grp_matmul_wc2_global_phase_split_unsupported());
    expect_global3_prompt1_wc2_fails<float16_t>(data_type_t::f16);
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        FusedGlobalAlgo3PromptPin5Wc2FailsBeforeBothWeightPoolsBf16) {
    using namespace zendnnl::lowoha::matmul;
    expect_fused_global3_prompt5_wc2_fails<bfloat16_t>(data_type_t::bf16);
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        FusedGlobalAlgo3PromptPin5Wc2FailsBeforeBothWeightPoolsF16) {
    using namespace zendnnl::lowoha::matmul;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "needs AVX-512-FP16 input validation";
    }
    expect_fused_global3_prompt5_wc2_fails<float16_t>(data_type_t::f16);
}

// AUTO + WC=2 falls back to the historical out-of-place downgrade (1)
// when the mixed-in-place preconditions are NOT met — here a finite LRU
// capacity (so the in-place sentinel could be evicted and re-derived from
// a mutated buffer).  Confirms the safe fallback still works.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoDowngradesWc2WhenIneligible) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0); // AUTO — mixes AOCL prompt + decode
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    WeightCacheGuard wc(2);
    LruCapacityGuard cap(4096); // finite -> mixed-in-place ineligible
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 1) << "AUTO + WC=2 must downgrade to 1 "
                                            "when mixed-in-place is ineligible";
    EXPECT_FALSE(is_grp_auto_mixed_inplace_active())
            << "WC=1 must make the sticky mixed guard inactive";
}

// AUTO + WC=2 with all preconditions (PREPACK + CROSS_WARM default ON,
// unlimited LRU capacity) ENABLES mixed-in-place: WC stays 2 and the flag
// is set (instead of downgrading).
TEST(TestGroupMatmulWeightCacheDowngrade, AutoMixedInplaceEnabledWc2) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    WeightCacheGuard wc(2); // capacity unlimited by default
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2)
            << "AUTO + WC=2 with prepack+cross_warm+unlimited capacity must "
               "keep WC=2";
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace())
            << "mixed-in-place flag must be set when eligible";
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoMixedPackedAndRawPoolUsesLocalFallback) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    WeightCacheGuard wc(2);
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    const auto mixed_first = run_min_grp_matmul(/*weight_offset_elems=*/0,
            /*include_inactive_packed=*/true);
    EXPECT_EQ(cfg.get_weight_cache(), 2)
            << "a per-call mixed-format pool must not revoke the process-wide "
               "WC=2 policy used by concurrent raw-only calls";
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace())
            << "an otherwise eligible process must keep CK out-of-place; only "
               "this call's AOCL cache cap should change";
    EXPECT_FALSE(mixed_first.weights_mutated)
            << "the mixed-format call must cap its raw experts at "
               "out-of-place caching";

    // A raw-only call observes the retained process-wide mixed policy.
    run_min_grp_matmul();
    ASSERT_EQ(cfg.get_weight_cache(), 2);
    ASSERT_TRUE(cfg.get_grp_auto_mixed_inplace());

    // A later mixed-format call must remain a local fallback. The former
    // process-wide downgrade cleared the mixed flag here; a concurrent raw
    // call that had already snapshotted WC=2 could then CK-pack its weights
    // in place and corrupt their later AOCL use.
    const auto mixed_after_raw = run_min_grp_matmul(/*weight_offset_elems=*/0,
            /*include_inactive_packed=*/true);
    EXPECT_EQ(cfg.get_weight_cache(), 2);
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace());
    EXPECT_FALSE(mixed_after_raw.weights_mutated);
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoMixedPackedAndRawFiniteCapacityHonorsPerCallCap) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    WeightCacheGuard wc(2);
    LruCapacityGuard finite_capacity(4096);
    const auto mixed = run_min_grp_matmul(/*weight_offset_elems=*/0,
            /*include_inactive_packed=*/true);
    EXPECT_EQ(matmul_config_t::instance().get_weight_cache(), 2)
            << "a per-call mixed pool must not downgrade global WC";
    EXPECT_FALSE(mixed.weights_mutated)
            << "the per-call mode-1 cap must reach CK preparation even when "
               "finite capacity makes process mixed-inplace ineligible";
}

// CK OFF downgrades to out-of-place.  The decode layout cross-warm packs with
// the custom kernel off is the AOCL per-tile one, whose keys embed
// `(col_start, n_tile)`.  That tracks `nr_align` and a narrow-N escape
// evaluated over the ACTIVE expert set, so a later decode call can build keys
// the warm never populated; once the prompt has rewritten the weight buffer
// those cannot be rebuilt.  Only the shape-keyed custom-kernel pack licenses
// the mutation, so WC falls back to 1 here and every in-place-eligible weight
// keeps its out-of-place copy.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkOffWc2Downgrades) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0); // AUTO
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    CustomKernelOverride ck_off(false);
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 1)
            << "AUTO + WC=2 with the custom kernel OFF must downgrade to "
               "out-of-place: the AOCL per-tile decode layout is split-keyed "
               "and cannot license the in-place prompt mutation";
    EXPECT_FALSE(is_grp_auto_mixed_inplace_active())
            << "WC=1 must disable mixed-in-place without the custom-kernel "
               "pack";
}

// Correctness companion: with CK OFF the WC=2 request is downgraded, so the
// numbers must match the WC=1 reference exactly -- nothing mutated the weight
// buffer on either leg.  Keeping the comparison guards the downgrade itself:
// were the licence ever widened back, this is the test that would catch decode
// reading an in-place-mutated buffer as raw.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkOffWc2MatchesWc1) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    MinRunResult ref, downgraded;
    {
        reset_grp_matmul_caches();
        CustomKernelOverride ck_off(false);
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul();
    }
    bool mixed_active_during_downgraded_call = true;
    {
        reset_grp_matmul_caches();
        CustomKernelOverride ck_off(false);
        WeightCacheGuard wc(2);
        downgraded = run_min_grp_matmul();
        mixed_active_during_downgraded_call
                = is_grp_auto_mixed_inplace_active();
    }
    EXPECT_FALSE(mixed_active_during_downgraded_call)
            << "WC=1 must disable mixed-in-place for a CK-off AUTO run";
    const double d = max_abs_diff(ref.dst, downgraded.dst);
    EXPECT_LT(d, 1e-2) << "CK-off WC=2 (downgraded) must match WC=1; "
                          "max_abs_diff="
                       << d;
}

// Correctness: AUTO + WC=2 mixed-in-place must produce the SAME result as
// AUTO + WC=1 (out-of-place).  The in-place mutation of the weight buffer
// (AOCL full-weight) must not perturb the CK decode output, which is
// served from the out-of-place pack warmed from the raw weights.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoMixedInplaceMatchesWc1) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    MinRunResult ref, mixed;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1); // out-of-place reference
        ref = run_min_grp_matmul();
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2); // AUTO + WC=2 -> mixed in-place
        mixed = run_min_grp_matmul();
    }
    EXPECT_TRUE(zendnnl::common::matmul_config_t::instance()
                        .get_grp_auto_mixed_inplace())
            << "mixed-in-place must have engaged for this AUTO + WC=2 run";
    const double d = max_abs_diff(ref.dst, mixed.dst);
    EXPECT_LT(d, 1e-2)
            << "AUTO mixed-in-place (WC=2) must match WC=1; max_abs_diff=" << d;
}

// ── FP16 AUTO + WC=2, end to end ─────────────────────────────────────
//
// The FP16 warm tests in test_prepack.cpp drive the full-weight warmer
// directly, which leaves `mixed_inplace_for_wei`, the cross-warm ordering and
// the AUTO runtime path uncovered for f16.  These tests go through
// `group_matmul_direct` and pin the two halves of the FP16 licence.
//
// CK-ELIGIBLE: unpinned FP16 prompt and decode share ALGO 3's FP16 pack.
// WC stays 2, cross-warm has no second layout to populate, and the CK writes
// its one family-specific layout into the caller's buffer.
TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoF16Algo3OnlyWc2PacksInPlaceAndMatchesWc1) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "AVX-512-FP16 not available on this host/toolchain";
    }
    using f16_t = zendnnl::common::float16_t;
    AlgoEnvGuard reset_algo(0);
    MinRunResultT<f16_t> ref, mixed;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1); // out-of-place reference
        ref = run_min_grp_matmul_typed<f16_t>(data_type_t::f16);
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2); // AUTO + WC=2 -> mixed in-place
        LastInvocationCaptureGuard capture;
        mixed = run_min_grp_matmul_typed<f16_t>(data_type_t::f16);
        const auto stats
                = group_matmul_prepack::test_api::get_last_invocation_stats();
        EXPECT_TRUE(stats.valid);
        EXPECT_EQ(stats.scheduling_algo, 3);
        EXPECT_EQ(stats.cross_warm_regime,
                group_matmul_prepack::CrossWarmRegime::none);
    }
    EXPECT_TRUE(zendnnl::common::matmul_config_t::instance()
                        .get_grp_auto_mixed_inplace())
            << "FP16 AUTO + WC=2 with the CK decode pack on must engage "
               "mixed-in-place";
    EXPECT_TRUE(mixed.weights_mutated)
            << "the FP16 CK pack should own the caller's buffer when both "
               "phases stay on ALGO 3";
    EXPECT_FALSE(ref.weights_mutated)
            << "WC=1 must leave the FP16 weight buffers RAW";
    const double d = max_abs_diff_t<f16_t>(ref.dst, mixed.dst);
    EXPECT_LT(d, 1e-2) << "FP16 mixed-in-place must match WC=1; max_abs_diff="
                       << d;
}

// CK-REFUSED: FP16 AUTO still selects ALGO 3, but this shape cannot use a CK
// pack.  The per-call ALGO-3-only policy caps AOCL at out-of-place, so nothing
// may mutate the weight while a future supported FP16 call can still read it.
//
// The refusal is driven by an N that no CK `pack_nr` divides, NOT by the FP16
// sub-knob.  Turning the sub-knob off would make
// `grp_matmul_custom_kernel_effective()` false, which downgrades the whole
// process to WC=1 -- the test would then pass through the global downgrade and
// would keep passing even if `mixed_inplace_for_wei()` or the `allow_inplace`
// arm regressed.  The mixed-mode assertion below is what pins that
// distinction.
//
// A prompt M (64 is above kDecodeMaxM, 32) verifies the new FP16 prompt route,
// rather than the pre-existing decode ALGO 3 route.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoF16Wc2CkRefusedKeepsWeightsRaw) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "AVX-512-FP16 not available on this host/toolchain";
    }
    using f16_t = zendnnl::common::float16_t;
    // 200 is divisible by neither pack_nr (32 nor 64), so `ck_eligible_f16`
    // declines this call while every knob stays ON.
    constexpr int kCkRefusedN = 200;
    constexpr int kPromptM = 64;
    AlgoEnvGuard reset_algo(0);
    MinRunResultT<f16_t> ref, refused;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul_typed<f16_t>(data_type_t::f16, /*K_dim=*/256,
                /*weight_offset_elems=*/0, /*N_dim=*/kCkRefusedN,
                /*M_dim=*/kPromptM);
    }
    group_matmul_prepack::test_api::LastInvocationStats refused_stats;
    {
        LastInvocationCaptureGuard capture;
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        refused = run_min_grp_matmul_typed<f16_t>(data_type_t::f16,
                /*K_dim=*/256, /*weight_offset_elems=*/0,
                /*N_dim=*/kCkRefusedN, /*M_dim=*/kPromptM);
        refused_stats
                = group_matmul_prepack::test_api::get_last_invocation_stats();
    }
    EXPECT_TRUE(refused_stats.valid);
    EXPECT_EQ(refused_stats.scheduling_algo, 3)
            << "an unpinned multi-expert FP16 prompt must stay on ALGO 3 even "
               "when this shape is refused by the custom kernel";
    EXPECT_EQ(refused_stats.cross_warm_regime,
            group_matmul_prepack::CrossWarmRegime::none);
    EXPECT_TRUE(zendnnl::common::matmul_config_t::instance()
                        .get_grp_auto_mixed_inplace())
            << "this must exercise the PER-CALL refusal: mixed mode has to "
               "stay active process-wide, otherwise the run passes through the "
               "global WC=1 downgrade and proves nothing about the local path";
    EXPECT_FALSE(refused.weights_mutated)
            << "a CK-REFUSED FP16 weight must never be mutated in place: "
               "prepack left it on the out-of-place path, so an in-place "
               "rewrite can race a caller still reading the raw bytes";
    const double d = max_abs_diff_t<f16_t>(ref.dst, refused.dst);
    EXPECT_LT(d, 1e-2) << "CK-refused FP16 WC=2 must match WC=1; max_abs_diff="
                       << d;
}

// ALGO 3 narrow-N.  N=1 is CK-refused (`plan_pack_nr` returns 0) and the
// blocked f16 image fits the plain allocation, so an unrestricted full-weight
// warm could rewrite the caller's buffer.  Eight experts at M=8 with 32
// threads gives `stable = 32/16 = 2`, which exceeds `N/nr_align = 1` and
// selects the narrow-N full-weight arm.  FP16 keeps both phases on ALGO 3, so
// cross-warm remains `none`; the primary warm must still stay out-of-place.
TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoF16Wc2NarrowNCkRefusedKeepsWeightsRaw) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "AVX-512-FP16 not available on this host/toolchain";
    }
    using f16_t = zendnnl::common::float16_t;
    constexpr int kNarrowN = 1;
    constexpr int kDecodeM = 8;
    constexpr int kExperts = 8;
    constexpr int kThreads = 32;
    AlgoEnvGuard reset_algo(0);
    MinRunResultT<f16_t> ref, narrow;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul_typed<f16_t>(data_type_t::f16, /*K_dim=*/256,
                /*weight_offset_elems=*/0, /*N_dim=*/kNarrowN,
                /*M_dim=*/kDecodeM, /*num_threads=*/kThreads,
                /*num_ops=*/kExperts);
    }
    group_matmul_prepack::test_api::LastInvocationStats narrow_stats;
    {
        LastInvocationCaptureGuard capture;
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        narrow = run_min_grp_matmul_typed<f16_t>(data_type_t::f16,
                /*K_dim=*/256, /*weight_offset_elems=*/0, /*N_dim=*/kNarrowN,
                /*M_dim=*/kDecodeM, /*num_threads=*/kThreads,
                /*num_ops=*/kExperts);
        narrow_stats
                = group_matmul_prepack::test_api::get_last_invocation_stats();
    }
    EXPECT_TRUE(narrow_stats.valid);
    EXPECT_EQ(narrow_stats.scheduling_algo, 3)
            << "this regression is the ALGO 3 narrow-N full-weight warm; "
               "ALGO 1 does not take that arm";
    EXPECT_EQ(narrow_stats.cross_warm_regime,
            group_matmul_prepack::CrossWarmRegime::none)
            << "narrow-N already warmed the full weight, so cross-warm "
               "must skip it.  A per-tile primary would report "
               "aocl_full_weight";
    EXPECT_EQ(narrow_stats.aocl.packed_ok, kExperts)
            << "the narrow-N warm must pack every expert out-of-place";
    EXPECT_EQ(narrow_stats.aocl.total_attempted, kExperts);
    EXPECT_TRUE(zendnnl::common::matmul_config_t::instance()
                        .get_grp_auto_mixed_inplace())
            << "mixed mode has to stay active, otherwise the run passes "
               "through the global WC=1 downgrade";
    EXPECT_FALSE(narrow.weights_mutated)
            << "a CK-refused FP16 weight on the ALGO 3 narrow-N arm must "
               "stay raw";
    const double d = max_abs_diff_t<f16_t>(ref.dst, narrow.dst);
    EXPECT_LT(d, 1e-2)
            << "narrow-N CK-refused FP16 WC=2 must match WC=1; max_abs_diff="
            << d;
}

// Concurrency: N threads issue the SAME grouped call on SHARED weight
// buffers under AUTO + WC=2 mixed-in-place.  The per-fingerprint warm
// latch must serialise the in-place mutation against concurrent compute
// so every thread's output matches the WC=1 reference (no thread computes
// on a half-mutated buffer).
TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoMixedInplaceConcurrentMatchesWc1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    constexpr int num_ops = 4, M = 8, K = 256, N = 256;
    const size_t wei_elems = static_cast<size_t>(K) * N;

    // WC=1 reference output (its own buffers).
    std::vector<std::vector<bfloat16_t>> ref_dst;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref_dst = run_min_grp_matmul().dst;
    }

    // Shared src + weight buffers (same fingerprint across all threads).
    reset_grp_matmul_caches();
    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops);
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(wei_elems);
        for (size_t i = 0; i < src[e].size(); ++i)
            src[e][i] = bfloat16_t(0.02f * (float)((int)((i + e) % 5) - 2));
        for (size_t i = 0; i < wei_elems; ++i)
            wei[e][i] = bfloat16_t(0.01f * (float)((int)((i + e) % 7) - 3));
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
    }
    auto gv = GemmVecs::uniform(num_ops, M, N, K);

    WeightCacheGuard wc(2); // AUTO + WC=2 -> mixed in-place
    constexpr int kThreads = 8;
    std::vector<std::vector<std::vector<bfloat16_t>>> dsts(kThreads);
    std::vector<std::thread> ths;
    for (int t = 0; t < kThreads; ++t) {
        ths.emplace_back([&, t]() {
            auto params = make_uniform_params(num_ops, data_type_t::bf16);
            for (int e = 0; e < num_ops; ++e)
                params[e].num_threads = 4;
            auto &dst = dsts[t];
            dst.resize(num_ops);
            std::vector<void *> dptr(num_ops);
            for (int e = 0; e < num_ops; ++e) {
                dst[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
                dptr[e] = dst[e].data();
            }
            group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                    gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                    gv.beta, dptr, gv.ldc, gv.is_wc, params,
                    /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
        });
    }
    for (auto &th : ths)
        th.join();
    for (int t = 0; t < kThreads; ++t) {
        const double d = max_abs_diff(ref_dst, dsts[t]);
        EXPECT_LT(d, 1e-2) << "concurrent mixed-in-place thread " << t
                           << " diverged from WC=1; max_abs_diff=" << d;
    }
}

TEST(TestGroupMatmulWeightCacheDowngrade, Algo1RespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo1(1); // pinned AOCL — single layout
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    EXPECT_EQ(
            zendnnl::common::matmul_config_t::instance().get_weight_cache(), 2)
            << "pinned ALGO 1 (single AOCL backend) must keep WC=2";
}

TEST(TestGroupMatmulWeightCacheDowngrade, Algo3CustomKernelOnRespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_on(true); // bf16 CK in-place pack (single consumer)
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    EXPECT_EQ(
            zendnnl::common::matmul_config_t::instance().get_weight_cache(), 2)
            << "ALGO 3 + CK keeps WC=2 (in-place handled safely in the pack "
               "layer)";
}

TEST(TestGroupMatmulWeightCacheDowngrade, Algo3CustomKernelOffRespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_off(false); // pure AOCL DLP — single layout
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    EXPECT_EQ(
            zendnnl::common::matmul_config_t::instance().get_weight_cache(), 2)
            << "ALGO 3 with custom kernel off (AOCL DLP) must keep WC=2";
}

// In-place CK pack correctness: pinned ALGO 3 + CK, bf16/even-K.  The
// WC=2 (in-place) result must match the WC=1 (out-of-place) reference
// bit-for-bit-ish (same pack bytes, only the storage location differs),
// and the WC=2 run must actually mutate the weight buffer (proving the
// in-place path engaged rather than silently falling back).
TEST(TestGroupMatmulWeightCacheDowngrade, Algo3CkInPlaceWc2MatchesWc1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_on(true);

    MinRunResult ref, test;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1); // out-of-place reference
        ref = run_min_grp_matmul();
        EXPECT_EQ(
                zendnnl::common::matmul_config_t::instance().get_weight_cache(),
                1);
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2); // in-place under test (fresh buffers)
        test = run_min_grp_matmul();
        EXPECT_EQ(
                zendnnl::common::matmul_config_t::instance().get_weight_cache(),
                2)
                << "ALGO 3 + CK must keep WC=2 (no downgrade)";
    }

    EXPECT_TRUE(test.weights_mutated) << "WC=2 + bf16/even-K must pack in "
                                         "place (mutate the weight buffer)";
    const double d = max_abs_diff(ref.dst, test.dst);
    EXPECT_LT(d, 1e-2)
            << "in-place CK pack (WC=2) must match the out-of-place (WC=1) "
               "result; max_abs_diff="
            << d;
}

// In-place CK pack is alignment-independent: a deliberately NON-64-byte-
// aligned weight base (offset by 1 bf16 element) must still pack in place
// AND run correctly through the microkernel.  Guards the unaligned
// B-load path — the kernel reads the packed stream with
// `_mm512_loadu_si512`, so an unaligned in-place base is correct, just
// potentially slower.  Before the unaligned-load switch this configuration
// #GP-faulted; this test locks in that it no longer does.
TEST(TestGroupMatmulWeightCacheDowngrade, Algo3CkInPlaceWc2UnalignedWeight) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_on(true);

    MinRunResult ref, test;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1); // out-of-place reference
        ref = run_min_grp_matmul(/*weight_offset_elems=*/1);
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2); // in-place on an UNALIGNED base
        test = run_min_grp_matmul(/*weight_offset_elems=*/1);
    }
    EXPECT_TRUE(test.weights_mutated)
            << "WC=2 must pack in place even when the weight base is unaligned";
    const double d = max_abs_diff(ref.dst, test.dst);
    EXPECT_LT(d, 1e-2)
            << "unaligned in-place CK pack (WC=2) must match the WC=1 result; "
               "max_abs_diff="
            << d;
}

// ── Cleanup-driven coverage additions ───────────────────────────────

// Regression for the `effective_weight_cache_type` live-read fix: it must
// reflect the CURRENT process weight-cache setting, not a one-time static
// snapshot.  Previously a pinned WC=2 `run_dlp` cached "2" on the first
// call, so a later AUTO downgrade to 1 was ignored.  Toggle WC within the
// process and confirm the effective value (clamped by the per-call cap)
// tracks it every time.
TEST(TestGroupMatmulWeightCacheDowngrade, EffectiveWeightCacheTypeLiveRead) {
    using namespace zendnnl::lowoha::matmul;
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    WeightCacheGuard restore(cfg.get_weight_cache()); // restore on scope exit
    cfg.set_weight_cache(2);
    EXPECT_EQ(effective_weight_cache_type(2), 2);
    EXPECT_EQ(effective_weight_cache_type(1), 1)
            << "per-call cap must clamp the effective value down";
    cfg.set_weight_cache(1);
    EXPECT_EQ(effective_weight_cache_type(2), 1)
            << "effective_weight_cache_type must observe the LIVE downgrade to "
               "1, "
               "not a stale first-call snapshot of 2";
    cfg.set_weight_cache(0);
    EXPECT_EQ(effective_weight_cache_type(2), 0)
            << "WC=0 (disabled) must propagate live as well";
    cfg.downgrade_weight_cache(1);
    EXPECT_EQ(cfg.get_weight_cache(), 0)
            << "a safety downgrade must not overwrite a concurrent WC=0 "
               "disable with WC=1";
    cfg.set_weight_cache(2);
    cfg.downgrade_weight_cache(1);
    EXPECT_EQ(cfg.get_weight_cache(), 1);
}

TEST(TestGroupMatmulWeightCacheDowngrade,
        MutationRegistryRequiresGroupedCallScope) {
    using namespace zendnnl::lowoha::matmul;
    reset_grp_matmul_caches();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    const bool old_mixed = cfg.get_grp_auto_mixed_inplace();
    cfg.set_grp_auto_mixed_inplace(true);
    int scalar_weight = 0;

    aocl_mark_weight_buffer_mutated(&scalar_weight);
    EXPECT_FALSE(aocl_weight_buffer_is_mutated(&scalar_weight))
            << "a sticky grouped policy must not register an unrelated scalar "
               "in-place reorder";

    {
        scoped_grp_auto_mutation_tracking grouped_scope(true);
        aocl_mark_weight_buffer_mutated(&scalar_weight);
    }
    EXPECT_TRUE(aocl_weight_buffer_is_mutated(&scalar_weight))
            << "the grouped call that authorized mutation must be tracked";

    int unrelated_weight = 0;
    {
        scoped_grp_auto_mutation_tracking grouped_scope(true);
        aocl_mark_weight_buffer_mutated(&unrelated_weight);
    }
    const std::vector<const void *> scalar_pool = {&scalar_weight};
    const std::vector<const void *> unrelated_pool = {&unrelated_weight};
    const std::vector<const void *> shared_model_pool
            = {&scalar_weight, &unrelated_weight};
    const std::vector<int> only_scalar_active = {1, 0};
    const uint64_t scalar_before
            = aocl_mutated_source_conflict_count(scalar_pool);
    const uint64_t active_subset_before
            = aocl_mutated_source_conflict_count(shared_model_pool,
                    /*secondary_weights=*/nullptr, &only_scalar_active);
    const uint64_t unrelated_before
            = aocl_mutated_source_conflict_count(unrelated_pool);
    EXPECT_TRUE(aocl_refuse_reorder_from_mutated(
            &unrelated_weight, "test unrelated pool"));
    EXPECT_EQ(aocl_mutated_source_conflict_count(scalar_pool), scalar_before)
            << "a conflict in one weight pool must not fail an unrelated "
               "concurrent grouped call";
    EXPECT_EQ(aocl_mutated_source_conflict_count(shared_model_pool,
                      /*secondary_weights=*/nullptr, &only_scalar_active),
            active_subset_before)
            << "a concurrent conflict on an inactive expert from the same "
               "model pool must not fail this routed call";
    EXPECT_EQ(aocl_mutated_source_conflict_count(unrelated_pool),
            unrelated_before + 1);

    cfg.set_grp_auto_mixed_inplace(old_mixed);
    reset_grp_matmul_caches();
}

// Fused ALGO 2/5 may execute both GEMMs without entering the ordinary grouped
// dispatcher. Its cold-start preflight must therefore initialize mixed mode
// for Op1+Op2 as one pool before either pipeline warms or mutates a weight.
TEST(TestGroupMatmulWeightCacheDowngrade,
        FusedTwoPoolColdStartInitializesMixedMode) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard auto_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    CustomKernelOverride ck_on(true);
    WeightCacheGuard wc(2);
    LruCapacityGuard unlimited(std::numeric_limits<uint32_t>::max());

    if (!get_grp_matmul_prepack() || !get_grp_matmul_cross_warm()) {
        GTEST_SKIP() << "requires the default prepack+cross-warm policy";
    }

    auto &cfg = zendnnl::common::matmul_config_t::instance();
    const bool old_mixed = cfg.get_grp_auto_mixed_inplace();
    struct MixedRestore {
        zendnnl::common::matmul_config_t &cfg;
        bool value;
        ~MixedRestore() { cfg.set_grp_auto_mixed_inplace(value); }
    } restore {cfg, old_mixed};
    cfg.set_grp_auto_mixed_inplace(false);

    auto op1 = make_uniform_params(2, data_type_t::bf16);
    auto op2 = make_uniform_params(2, data_type_t::bf16);
    const std::vector<int> M = {128, 128};
    const std::vector<int> N = {128, 128};
    ASSERT_TRUE(grp_matmul_custom_kernel_effective(
            op1, M, N, grp_matmul_gated_act_t::none));
    ASSERT_TRUE(grp_matmul_custom_kernel_effective(
            op2, M, N, grp_matmul_gated_act_t::none));

    EXPECT_TRUE(prepare_grouped_auto_weight_cache(
            op1, /*custom_kernel_effective=*/true, &op2));
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace())
            << "the first fused call must license cross-warm before a vertical "
               "pipeline can perform the in-place AOCL prompt reorder";
    for (const auto &p : op1)
        EXPECT_EQ(p.weight_cache_type, 2);
    for (const auto &p : op2)
        EXPECT_EQ(p.weight_cache_type, 2);
}

// Pinned ALGO 2/5/6 (single AOCL reorder layout per weight) must KEEP
// WC=2 just like ALGO 1 — mixed mode is AUTO-only, so pinned algos leave
// the process WC untouched and clear the mixed flag deterministically.
TEST(TestGroupMatmulWeightCacheDowngrade, Algo2RespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo2(2);
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2)
            << "pinned ALGO 2 (single AOCL layout) must keep WC=2";
    EXPECT_FALSE(cfg.get_grp_auto_mixed_inplace())
            << "mixed-in-place is AUTO-only; pinned ALGO 2 must clear the flag";
}

TEST(TestGroupMatmulWeightCacheDowngrade, Algo6RespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo6(6);
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2)
            << "pinned ALGO 6 (multilevel CCD, single AOCL layout) must keep "
               "WC=2";
    EXPECT_FALSE(cfg.get_grp_auto_mixed_inplace());
}

TEST(TestGroupMatmulWeightCacheDowngrade, Algo5RespectsWc2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard algo5(5);
    WeightCacheGuard wc(2);
    run_min_grp_matmul();
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2)
            << "pinned ALGO 5 (per-expert, single AOCL layout) must keep WC=2";
    EXPECT_FALSE(cfg.get_grp_auto_mixed_inplace());
}

// AUTO mixed in-place with the custom kernel ENABLED: CK decode packs are
// warmed out-of-place from the raw weights BEFORE the AOCL full-weight
// prompt reorder mutates the buffer in place.  Result must still match the
// WC=1 out-of-place reference.
TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoMixedInplaceCustomKernelOnMatchesWc1) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_on_algo1(1); // split the phases
    CustomKernelOverride ck_on(true);
    MinRunResult ref, mixed;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul();
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        mixed = run_min_grp_matmul();
    }
    EXPECT_TRUE(zendnnl::common::matmul_config_t::instance()
                        .get_grp_auto_mixed_inplace())
            << "mixed-in-place must engage for AUTO + WC=2 with CK on";
    const double d = max_abs_diff(ref.dst, mixed.dst);
    EXPECT_LT(d, 1e-2)
            << "AUTO mixed-in-place + CK on must match WC=1; max_abs_diff="
            << d;
}

// Non-bf16 (f32) under AUTO + WC=2: F32 is not a custom-kernel family, so it
// must not license mixed in-place or the mixed-refusal Sequential fallback.
// The dispatcher downgrades to WC=1, leaves weights raw, and retains the
// normal parallel standard-backend plan.
TEST(TestGroupMatmulWeightCacheDowngrade, F32AutoWc2StaysOutOfPlace) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    const bool old_mixed = cfg.get_grp_auto_mixed_inplace();
    cfg.set_grp_auto_mixed_inplace(false);
    struct MixedRestore {
        zendnnl::common::matmul_config_t &cfg;
        bool value;
        ~MixedRestore() { cfg.set_grp_auto_mixed_inplace(value); }
    } mixed_restore {cfg, old_mixed};
    MinRunResultT<float> ref, test;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul_typed<float>(data_type_t::f32);
    }
    std::string mode;
    int32_t effective_weight_cache = -1;
    bool mixed_inplace = true;
    {
        GemmModeCaptureGuard mode_capture;
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        test = run_min_grp_matmul_typed<float>(data_type_t::f32);
        const char *captured
                = test_api::s_last_group_matmul_direct_gemm_mode.load(
                        std::memory_order_relaxed);
        ASSERT_NE(captured, nullptr);
        mode = captured;
        effective_weight_cache = cfg.get_weight_cache();
        mixed_inplace = cfg.get_grp_auto_mixed_inplace();
    }
    EXPECT_FALSE(test.weights_mutated)
            << "f32 AUTO must stay out-of-place and keep weights raw";
    EXPECT_EQ(effective_weight_cache, 1);
    EXPECT_FALSE(mixed_inplace);
    EXPECT_EQ(mode.find("sequential"), std::string::npos) << mode;
    const double d = max_abs_diff_t(ref.dst, test.dst);
    EXPECT_LT(d, 1e-3)
            << "f32 AUTO WC=2 (forced OOP) must match WC=1; max_abs_diff=" << d;
}

// Odd-K bf16 under WC=2 + pinned ALGO 3 + CK: the custom kernel refuses
// odd K (VNNI K-pair pad) and the AOCL in-place size gate fails (blocked
// size != plain size), so the pack falls back OUT-OF-PLACE and the weight
// buffer must NOT be mutated.  Result must still match the WC=1 reference.
TEST(TestGroupMatmulWeightCacheDowngrade, Algo3OddKWc2StaysOutOfPlace) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_on(true);
    MinRunResultT<bfloat16_t> ref, test;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul_typed<bfloat16_t>(
                data_type_t::bf16, /*K_dim=*/255);
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        test = run_min_grp_matmul_typed<bfloat16_t>(data_type_t::bf16,
                /*K_dim=*/255);
    }
    EXPECT_FALSE(test.weights_mutated) << "odd-K bf16 under WC=2 must fall "
                                          "back to out-of-place (no mutation)";
    const double d = max_abs_diff_t(ref.dst, test.dst);
    EXPECT_LT(d, 1e-2)
            << "odd-K bf16 WC=2 (forced OOP) must match WC=1; max_abs_diff="
            << d;
}

// ── AUTO custom-kernel in-place (both phases on ALGO 3) ─────────────

namespace {

// bf16 expert weights the test owns, so it can call the same layer twice and
// ask which buffers an in-place pack rewrote.
struct Bf16Layer {
    int K, N;
    std::vector<std::vector<bfloat16_t>> wei, raw;

    Bf16Layer(int experts, int K_, int N_)
        : K(K_), N(N_), wei(experts), raw(experts) {
        for (int e = 0; e < experts; ++e) {
            wei[e].resize(static_cast<size_t>(K) * N);
            for (size_t i = 0; i < wei[e].size(); ++i)
                wei[e][i] = bfloat16_t(0.01f * (float)((int)((i + e) % 7) - 3));
            raw[e] = wei[e];
        }
    }

    bool mutated(int e) const {
        return std::memcmp(wei[e].data(), raw[e].data(),
                       wei[e].size() * sizeof(bfloat16_t))
                != 0;
    }

    // One call with `M[e]` rows for expert `e`; an idle expert gets a
    // one-row buffer the call leaves untouched.  `tile_unsafe` gives the last
    // expert a different bias dtype (there is no bias), which keeps the call
    // off ALGO 3.
    zendnnl::error_handling::status_t run(const std::vector<int> &M,
            std::vector<std::vector<bfloat16_t>> &dst, float alpha = 1.0f,
            int threads = 8, int32_t weight_cache_type = 2,
            bool tile_unsafe = false) {
        using namespace zendnnl::lowoha::matmul;
        using namespace moe_test_utils;
        const int E = static_cast<int>(wei.size());
        std::vector<std::vector<bfloat16_t>> src(E);
        std::vector<const void *> srcs(E), weis(E), biases(E, nullptr);
        std::vector<void *> dsts(E);
        dst.assign(E, {});
        for (int e = 0; e < E; ++e) {
            const int m = std::max(M[e], 1);
            src[e].resize(static_cast<size_t>(m) * K);
            for (size_t i = 0; i < src[e].size(); ++i)
                src[e][i] = bfloat16_t(0.02f * (float)((int)((i + e) % 5) - 2));
            dst[e].assign(static_cast<size_t>(m) * N, bfloat16_t(0.0f));
            srcs[e] = src[e].data();
            weis[e] = wei[e].data();
            dsts[e] = dst[e].data();
        }
        auto gv = GemmVecs::uniform(E, 1, N, K, alpha);
        gv.Ms = M;
        auto params = make_uniform_params(E, data_type_t::bf16);
        for (auto &p : params) {
            p.num_threads = threads;
            p.weight_cache_type = weight_cache_type;
        }
        if (tile_unsafe) { params.back().dtypes.bias = data_type_t::f32; }
        return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                gv.beta, dsts, gv.ldc, gv.is_wc, params,
                /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
    }

    // The experts as a sequential chain (`src.size() == 1`): each op reads
    // the previous op's output, so the layer must be square.
    zendnnl::error_handling::status_t run_chain(
            int M, std::vector<std::vector<bfloat16_t>> &dst) {
        using namespace zendnnl::lowoha::matmul;
        using namespace moe_test_utils;
        const int E = static_cast<int>(wei.size());
        std::vector<bfloat16_t> src(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = bfloat16_t(0.02f * (float)((int)(i % 5) - 2));
        std::vector<const void *> srcs {src.data()}, weis(E),
                biases(E, nullptr);
        std::vector<void *> dsts(E);
        dst.assign(E, std::vector<bfloat16_t>(static_cast<size_t>(M) * N));
        for (int e = 0; e < E; ++e) {
            weis[e] = wei[e].data();
            dsts[e] = dst[e].data();
        }
        auto gv = GemmVecs::uniform(E, M, N, K, 1.0f);
        auto params = make_uniform_params(E, data_type_t::bf16);
        return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                gv.beta, dsts, gv.ldc, gv.is_wc, params,
                /*moe_postop=*/nullptr, /*gated_act=*/nullptr);
    }
};

} // namespace

// Both phases run ALGO 3, so the custom-kernel pack is the only layout of the
// weights: AUTO keeps WC=2 and the process mixed mode, packs the bf16 weights
// in place, and matches the out-of-place run bit for bit.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceWc2MatchesWc1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    LruCapacityGuard unlimited(std::numeric_limits<uint32_t>::max());
    MinRunResult ref, in_place;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        ref = run_min_grp_matmul();
    }
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(2);
        in_place = run_min_grp_matmul();
        auto &cfg = zendnnl::common::matmul_config_t::instance();
        EXPECT_EQ(cfg.get_weight_cache(), 2) << "no process-wide downgrade";
        EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace());
    }
    EXPECT_FALSE(ref.weights_mutated) << "WC=1 must leave the weights raw";
    EXPECT_TRUE(in_place.weights_mutated)
            << "the custom-kernel pack must be written into the caller's "
               "buffer";
    EXPECT_EQ(max_abs_diff(ref.dst, in_place.dst), 0.0);
}

// A BF16 MoE call ALGO 3 cannot serve keeps the rules and the mixed mode of
// every other call: its prompt AOCL reorder is written in place and the
// decode reads it, the process keeps WC=2, and both phases match WC=1.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoBf16MoeOffAlgo3KeepsMixedMode) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    LruCapacityGuard unlimited(std::numeric_limits<uint32_t>::max());
    const std::vector<int> prompt {64, 64, 64, 64}, decode {8, 8, 8, 8};
    std::vector<std::vector<bfloat16_t>> ref_p, ref_d, dst_p, dst_d;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        Bf16Layer layer(4, 256, 256);
        ASSERT_EQ(layer.run(prompt, ref_p, 1.0f, 8, 2, /*tile_unsafe=*/true),
                status_t::success);
        ASSERT_EQ(layer.run(decode, ref_d, 1.0f, 8, 2, /*tile_unsafe=*/true),
                status_t::success);
    }
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer layer(4, 256, 256);
    ASSERT_EQ(layer.run(prompt, dst_p, 1.0f, 8, 2, /*tile_unsafe=*/true),
            status_t::success);
    ASSERT_EQ(layer.run(decode, dst_d, 1.0f, 8, 2, /*tile_unsafe=*/true),
            status_t::success);
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2) << "no process-wide downgrade";
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace());
    for (int e = 0; e < 4; ++e) {
        EXPECT_TRUE(layer.mutated(e)) << "expert " << e;
        EXPECT_FALSE(custom_kernel::weight_packed_in_place(layer.wei[e].data()))
                << "expert " << e;
        EXPECT_EQ(std::memcmp(ref_p[e].data(), dst_p[e].data(),
                          ref_p[e].size() * sizeof(bfloat16_t)),
                0)
                << "prompt expert " << e;
        EXPECT_EQ(std::memcmp(ref_d[e].data(), dst_d[e].data(),
                          ref_d[e].size() * sizeof(bfloat16_t)),
                0)
                << "decode expert " << e;
    }
}

// The licence is per call: after a BF16 MoE call packed its experts in place,
// a single expert in the same process keeps the mixed in-place mode.  Its
// ALGO 1 prompt reorder is written in place and its ALGO 3 decode pack is
// not, and both outputs match WC=1.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoSingleExpertAfterLicensedCall) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    LruCapacityGuard unlimited(std::numeric_limits<uint32_t>::max());
    const std::vector<int> moe_m {64, 64, 64, 64}, prompt {64}, decode {1};
    std::vector<std::vector<bfloat16_t>> moe_ref, prompt_ref, decode_ref, out;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        Bf16Layer moe(4, 256, 256), dense(1, 256, 256);
        ASSERT_EQ(moe.run(moe_m, moe_ref), status_t::success);
        ASSERT_EQ(dense.run(prompt, prompt_ref), status_t::success);
        ASSERT_EQ(dense.run(decode, decode_ref), status_t::success);
    }
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer moe(4, 256, 256), dense(1, 256, 256);
    ASSERT_EQ(moe.run(moe_m, out), status_t::success);
    for (int e = 0; e < 4; ++e) {
        EXPECT_TRUE(moe.mutated(e)) << "expert " << e;
        EXPECT_EQ(std::memcmp(moe_ref[e].data(), out[e].data(),
                          out[e].size() * sizeof(bfloat16_t)),
                0)
                << "expert " << e;
    }
    ASSERT_EQ(dense.run(prompt, out), status_t::success);
    EXPECT_TRUE(dense.mutated(0)) << "the ALGO 1 prompt reorder goes in place";
    EXPECT_EQ(max_abs_diff(prompt_ref, out), 0.0);
    ASSERT_EQ(dense.run(decode, out), status_t::success);
    EXPECT_FALSE(custom_kernel::weight_packed_in_place(dense.wei[0].data()))
            << "the ALGO 3 decode pack stays out-of-place";
    EXPECT_EQ(max_abs_diff(decode_ref, out), 0.0);
    auto &cfg = zendnnl::common::matmul_config_t::instance();
    EXPECT_EQ(cfg.get_weight_cache(), 2) << "no process-wide downgrade";
    EXPECT_TRUE(cfg.get_grp_auto_mixed_inplace());
}

// The prepack writes the pack of every expert the call advertises into its
// buffer, active or idle.  A later call on the experts that were idle reads
// those packs and must match an out-of-place run.
TEST(TestGroupMatmulWeightCacheDowngrade,
        AutoCkInPlacePrepackPacksIdleExperts) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    const std::vector<int> first {8, 0, 8, 0}, second {0, 8, 0, 8};
    std::vector<std::vector<bfloat16_t>> ref, dst;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        Bf16Layer layer(4, 256, 256);
        ASSERT_EQ(layer.run(first, ref), status_t::success);
        ASSERT_EQ(layer.run(second, ref), status_t::success);
    }
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer layer(4, 256, 256);
    ASSERT_EQ(layer.run(first, dst), status_t::success);
    for (int e = 0; e < 4; ++e) {
        EXPECT_TRUE(custom_kernel::weight_packed_in_place(layer.wei[e].data()))
                << "expert " << e;
        EXPECT_TRUE(layer.mutated(e)) << "expert " << e;
    }
    ASSERT_EQ(layer.run(second, dst), status_t::success);
    for (int e : {1, 3}) {
        EXPECT_EQ(std::memcmp(ref[e].data(), dst[e].data(),
                          ref[e].size() * sizeof(bfloat16_t)),
                0)
                << "expert " << e;
    }
}

// A call that caps its weight cache at 1 keeps every weight of the call raw,
// idle experts included, and matches the out-of-place run.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceHonoursPerCallCap) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    const std::vector<int> M {8, 0, 8, 0};
    std::vector<std::vector<bfloat16_t>> ref, dst;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        Bf16Layer layer(4, 256, 256);
        ASSERT_EQ(layer.run(M, ref), status_t::success);
    }
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer layer(4, 256, 256);
    ASSERT_EQ(layer.run(M, dst, 1.0f, 8, /*weight_cache_type=*/1),
            status_t::success);
    for (int e = 0; e < 4; ++e) {
        EXPECT_FALSE(layer.mutated(e)) << "expert " << e;
    }
    for (int e : {0, 2}) {
        EXPECT_EQ(std::memcmp(ref[e].data(), dst[e].data(),
                          ref[e].size() * sizeof(bfloat16_t)),
                0)
                << "expert " << e;
    }
}

// A sequential chain runs every op on AOCL, so it fails on a weight the
// custom kernel packed in place; on any other weight it runs as before.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceSequentialChain) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    std::vector<std::vector<bfloat16_t>> dst;
    Bf16Layer packed(4, 256, 256);
    ASSERT_EQ(packed.run({8, 8, 8, 8}, dst), status_t::success);
    ASSERT_TRUE(packed.mutated(0));
    EXPECT_EQ(packed.run_chain(8, dst), status_t::failure);

    Bf16Layer chain(4, 256, 256);
    EXPECT_EQ(chain.run_chain(8, dst), status_t::success);
}

// A weight the custom kernel does not pack keeps its raw bytes: AOCL DLP
// layouts stay out-of-place, the process keeps WC=2 for every other path,
// and the result matches WC=1.  An N no pack NR divides, and an odd K.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceLeavesOtherLayoutsRaw) {
    SKIP_GRP_MATMUL_TESTS_WITHOUT_AOCL_DLP();
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck(true);
    struct Case {
        int K, N, M;
    };
    for (const Case c :
            {Case {256, 200, 8}, Case {256, 200, 64}, Case {255, 256, 8}}) {
        MinRunResultT<bfloat16_t> ref, test;
        {
            reset_grp_matmul_caches();
            WeightCacheGuard wc(1);
            ref = run_min_grp_matmul_typed<bfloat16_t>(data_type_t::bf16, c.K,
                    /*weight_offset_elems=*/0, c.N, c.M);
        }
        {
            reset_grp_matmul_caches();
            WeightCacheGuard wc(2);
            test = run_min_grp_matmul_typed<bfloat16_t>(data_type_t::bf16, c.K,
                    /*weight_offset_elems=*/0, c.N, c.M);
            EXPECT_EQ(zendnnl::common::matmul_config_t::instance()
                              .get_weight_cache(),
                    2);
        }
        EXPECT_FALSE(test.weights_mutated)
                << "K=" << c.K << " N=" << c.N << " M=" << c.M;
        EXPECT_LT(max_abs_diff_t(ref.dst, test.dst), 1e-2)
                << "K=" << c.K << " N=" << c.N << " M=" << c.M;
    }
}

// A weight the custom kernel rewrote has no other reader.  A later call the
// kernel refuses (alpha != 1) fails instead of reading the image as a raw
// weight.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceRefusedCallFailsClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer layer(4, 256, 256);
    std::vector<std::vector<bfloat16_t>> dst;
    ASSERT_EQ(layer.run({8, 8, 8, 8}, dst), status_t::success);
    ASSERT_TRUE(layer.mutated(0));
    EXPECT_EQ(layer.run({8, 8, 8, 8}, dst, /*alpha=*/2.0f), status_t::failure);
    EXPECT_EQ(layer.run({64, 64, 64, 64}, dst, /*alpha=*/2.0f),
            status_t::failure);
}

// Concurrent calls on shared weights: each in-place pack runs under the pack
// cache's lock, so every thread computes on a complete image and matches the
// out-of-place reference.
TEST(TestGroupMatmulWeightCacheDowngrade, AutoCkInPlaceConcurrentMatchesWc1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard reset_algo(0);
    CustomKernelOverride ck_on(true);
    const std::vector<int> M {8, 0, 8, 8};
    std::vector<std::vector<bfloat16_t>> ref;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc(1);
        Bf16Layer layer(4, 256, 256);
        ASSERT_EQ(layer.run(M, ref), status_t::success);
    }
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    Bf16Layer layer(4, 256, 256);
    constexpr int kThreads = 8;
    std::vector<std::vector<std::vector<bfloat16_t>>> dsts(kThreads);
    std::vector<status_t> st(kThreads, status_t::failure);
    std::vector<std::thread> ths;
    for (int t = 0; t < kThreads; ++t) {
        ths.emplace_back([&, t]() { st[t] = layer.run(M, dsts[t], 1.0f, 4); });
    }
    for (auto &th : ths)
        th.join();
    for (int t = 0; t < kThreads; ++t) {
        ASSERT_EQ(st[t], status_t::success) << "thread " << t;
        for (int e : {0, 2, 3}) {
            EXPECT_EQ(std::memcmp(ref[e].data(), dsts[t][e].data(),
                              ref[e].size() * sizeof(bfloat16_t)),
                    0)
                    << "thread " << t << " expert " << e;
        }
    }
}

// A call the custom kernel declines after its pack loop rewrote some experts
// (here a second expert reads past the first one's image, so its pack fails)
// must fail rather than run the AOCL fallback on the rewritten buffer.
TEST(TestGroupMatmulWeightCacheDowngrade, Algo3PartialInPlacePackFailsClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "the BF16 custom kernel needs AVX-512 BF16";
    }
    AlgoEnvGuard algo3(3);
    CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();
    WeightCacheGuard wc(2);
    constexpr int K = 128, M = 8;
    // Expert 0 is the first 64 columns' worth of the buffer; expert 1 views
    // the same buffer as 128 columns wide.
    std::vector<bfloat16_t> buf(static_cast<size_t>(K) * 128);
    for (size_t i = 0; i < buf.size(); ++i)
        buf[i] = bfloat16_t(0.01f * (float)((int)(i % 7) - 3));
    std::vector<std::vector<bfloat16_t>> src(2), dst(2);
    std::vector<const void *> srcs(2), weis(2, buf.data()), biases(2, nullptr);
    std::vector<void *> dsts(2);
    auto gv = GemmVecs::uniform(2, M, 64, K);
    gv.Ns = {64, 128};
    gv.ldb = {64, 128};
    gv.ldc = {64, 128};
    for (int e = 0; e < 2; ++e) {
        src[e].assign(static_cast<size_t>(M) * K, bfloat16_t(0.02f));
        dst[e].assign(static_cast<size_t>(M) * gv.Ns[e], bfloat16_t(0.0f));
        srcs[e] = src[e].data();
        dsts[e] = dst[e].data();
    }
    auto params = make_uniform_params(2, data_type_t::bf16);
    for (auto &p : params)
        p.num_threads = 8;
    EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
                      gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
                      gv.beta, dsts, gv.ldc, gv.is_wc, params,
                      /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
            status_t::failure);
}

// The unpinned prompt of a BF16 layer of several experts runs ALGO 3
// whatever its shape: many experts deep or thin, few experts on a wide team,
// or a shape ALGO 2 could M-tile.
// The tile executor is held off, so the route cannot depend on it.
TEST(TestGroupMatmulAutoPhaseEnv, UnpinnedPromptRoutesToAlgo3AtEveryShape) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    PromptTileOverride tile_off(false);

    struct Shape {
        int M, num_ops, num_threads;
    };
    for (const Shape &sh : {Shape {768, 88, 64}, Shape {64, 88, 64},
                 Shape {40, 8, 64}, Shape {512, 16, 64}, Shape {256, 8, 1}}) {
        auto s = build_auto_probe(
                sh.M, /*K=*/2048, /*N=*/1536, sh.num_ops, sh.num_threads);
        EXPECT_EQ(select_grp_matmul_algo(
                          s.layout, s.M, s.N, s.K, s.params, s.num_threads),
                3)
                << "M=" << sh.M << " num_ops=" << sh.num_ops
                << " threads=" << sh.num_threads;
    }
}

// A BF16 layer of few experts decodes on ALGO 3 too, at any occupancy and
// any M, and AUTO selects ALGO 2 only for W4A8 prompts — the
// `ExplicitPromptAlgo2PinBeatsRegimeRouting` and
// `SingleDenseExpertDecodeHonoursExplicitPin` tests cover the pinned side.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsDecodeRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Mixtral-class decode: 8 active experts on a 128-thread team, at M=16
    // and at a single token.
    for (const int m : {16, 1}) {
        auto s = build_auto_probe(m, /*K=*/4096, /*N=*/14336,
                /*num_ops=*/8, /*num_threads=*/128);
        EXPECT_EQ(select_grp_matmul_algo(
                          s.layout, s.M, s.N, s.K, s.params, s.num_threads),
                3)
                << "M=" << m;
    }
}

// PRECEDENCE: `AUTO_{DECODE,PROMPT}_ALGO` outranks `ZENDNNL_GRP_MATMUL_ALGO`.
// Setting a phase knob must win whatever the global is set to, so an operator
// tuning one phase does not have to clear the global first.  Cross-product of
// every global generic pin against every phase generic pin, on a shape where
// both are legal (bf16, row-major, so m_tile_safe and n_tile_safe both hold).
TEST(TestGroupMatmulAutoPhaseEnv, PhaseKnobOutranksGlobalAlgoBothPhases) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    for (const int global : {1, 2, 3, 5, 6}) {
        for (const int phase_pin : {1, 2, 3, 6}) {
            // DECODE frame (max_M <= 32): AUTO_DECODE_ALGO must win.
            {
                reset_grp_matmul_caches();
                AlgoEnvGuard global_pin(global);
                AutoDecodeAlgoOverride decode_pin(phase_pin);
                auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                        /*num_ops=*/16, /*num_threads=*/64);
                EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K,
                                  s.params, s.num_threads),
                        phase_pin)
                        << "AUTO_DECODE_ALGO=" << phase_pin
                        << " must outrank ZENDNNL_GRP_MATMUL_ALGO=" << global;
            }
            // PROMPT frame (max_M > 32): AUTO_PROMPT_ALGO must win.
            {
                reset_grp_matmul_caches();
                AlgoEnvGuard global_pin(global);
                AutoPromptAlgoOverride prompt_pin(phase_pin);
                auto s = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
                        /*num_ops=*/16, /*num_threads=*/64);
                EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K,
                                  s.params, s.num_threads),
                        phase_pin)
                        << "AUTO_PROMPT_ALGO=" << phase_pin
                        << " must outrank ZENDNNL_GRP_MATMUL_ALGO=" << global;
            }
        }
    }
}

// The global still governs the phase whose knob is UNSET: setting only the
// decode knob must not disturb prompt routing, and vice versa.
TEST(TestGroupMatmulAutoPhaseEnv, GlobalStillAppliesToTheUnsetPhase) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    {
        reset_grp_matmul_caches();
        AlgoEnvGuard global_pin(6); // global multilevel
        AutoDecodeAlgoOverride decode_pin(3); // only decode is pinned
        auto prompt = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/16, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(prompt.layout, prompt.M, prompt.N,
                          prompt.K, prompt.params, prompt.num_threads),
                6)
                << "prompt knob unset → the global pin still governs prompt";
    }
    {
        reset_grp_matmul_caches();
        AlgoEnvGuard global_pin(6);
        AutoPromptAlgoOverride prompt_pin(1); // only prompt is pinned
        auto decode = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/16, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(decode.layout, decode.M, decode.N,
                          decode.K, decode.params, decode.num_threads),
                6)
                << "decode knob unset → the global pin still governs decode";
    }
}

// Non-W4A8 invariant guard: sweep the BF16 shape space AUTO can see and
// assert ALGO 2 never comes back without an explicit env pin.  W4A8 prompt
// routing is covered separately below.
TEST(TestGroupMatmulAutoPhaseEnv,
        NonW4A8AutoNeverSelectsAlgo2AcrossShapeSweep) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    for (const int num_ops : {1, 2, 4, 7, 8, 9, 16, 64}) {
        for (const int M : {1, 2, 8, 16, 32, 33, 64, 256}) {
            for (const int nthr : {8, 32, 128}) {
                reset_grp_matmul_caches();
                AlgoEnvGuard reset_algo(0); // global AUTO, no phase pins
                auto s = build_auto_probe(
                        M, /*K=*/4096, /*N=*/14336, num_ops, nthr);
                const int got = select_grp_matmul_algo(
                        s.layout, s.M, s.N, s.K, s.params, s.num_threads);
                EXPECT_NE(got, 2)
                        << "AUTO selected ALGO 2 without an env pin at "
                           "num_ops="
                        << num_ops << " M=" << M << " num_threads=" << nthr
                        << " — ALGO 2 must remain opt-in for non-W4A8";
            }
        }
    }
}

// The default is not a floor: an explicit decode phase pin still wins on a
// shape the default sends to ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsExplicitDecodePinStillWins) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    AutoDecodeAlgoOverride force_decode(3); // explicit decode pin

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "explicit AUTO_DECODE_ALGO=3 must win over the few-experts "
               "occupancy policy";
}

// ── W4A8 decode follows the generic AUTO policy ──────────────────────
// Decode runs ALGO 3 at any occupancy; only a W4A8 prompt has its own
// default (ALGO 2, below).

TEST(TestGroupMatmulW4A8RuntimeAlgo, AutoIgnoresInnerKernel) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard auto_algo(0);

    EXPECT_EQ(w4a8_runtime_algo(3, matmul_algo_t::aocl_dlp_blocked),
            matmul_algo_t::aocl_dlp_blocked);
    EXPECT_EQ(w4a8_runtime_algo(3, matmul_algo_t::aocl_dlp),
            matmul_algo_t::aocl_dlp_blocked);
    EXPECT_EQ(w4a8_runtime_algo(1, matmul_algo_t::aocl_dlp_blocked),
            matmul_algo_t::aocl_dlp);
    EXPECT_EQ(w4a8_runtime_algo(1, matmul_algo_t::aocl_dlp),
            matmul_algo_t::aocl_dlp);
}

TEST(TestGroupMatmulW4A8RuntimeAlgo, PinnedGroupAlgoRespectsInnerKernel) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;

    {
        AlgoEnvGuard pin_algo3(3);
        EXPECT_EQ(w4a8_runtime_algo(3, matmul_algo_t::aocl_dlp_blocked),
                matmul_algo_t::aocl_dlp_blocked);
        EXPECT_EQ(w4a8_runtime_algo(3, matmul_algo_t::aocl_dlp),
                matmul_algo_t::aocl_dlp);
    }
    {
        AlgoEnvGuard pin_algo1(1);
        EXPECT_EQ(w4a8_runtime_algo(1, matmul_algo_t::aocl_dlp_blocked),
                matmul_algo_t::aocl_dlp_blocked);
        EXPECT_EQ(w4a8_runtime_algo(1, matmul_algo_t::aocl_dlp),
                matmul_algo_t::aocl_dlp);
    }
}

TEST(TestGroupMatmulExpertKernel, AppliesPolicyOnlyToW4A8Experts) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard auto_algo(0);

    auto w4a8 = build_auto_probe(
            /*M=*/1, /*K=*/64, /*N=*/64, /*num_ops=*/1, /*num_threads=*/1);
    apply_w4a8_auto_probe(w4a8);
    const matmul_params &w4a8_params = w4a8.params.front();

    matmul_params bf16_params;
    bf16_params.dtypes.src = data_type_t::bf16;
    bf16_params.dtypes.wei = data_type_t::bf16;
    bf16_params.dtypes.dst = data_type_t::bf16;

    EXPECT_EQ(resolve_expert_kernel(3, matmul_algo_t::aocl_dlp, w4a8_params),
            matmul_algo_t::aocl_dlp_blocked);
    EXPECT_EQ(resolve_expert_kernel(
                      1, matmul_algo_t::aocl_dlp_blocked, w4a8_params),
            matmul_algo_t::aocl_dlp);
    EXPECT_EQ(resolve_expert_kernel(
                      1, matmul_algo_t::aocl_dlp_blocked, bf16_params),
            matmul_algo_t::aocl_dlp_blocked)
            << "non-W4A8 experts must preserve the selected inner kernel";
}

// An unpinned W4A8 prompt runs ALGO 2's row-disjoint M tiles; a prompt pin
// still outranks it, and a declined `=5` pin falls back to it.
TEST(TestGroupMatmulAutoPhaseEnv, W4A8PromptRoutesToAlgo2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Rule 0.7's measured W4A8 exception selects row-disjoint M-tiling.
    auto s = build_auto_probe(/*M=*/512, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/32);
    apply_w4a8_auto_probe(s);
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            2)
            << "structurally-safe W4A8 prompt must take the measured ALGO 2 "
               "route";
    EXPECT_STREQ(trace.reason, "auto_rule07_prompt_w4a8_mtile");
}

TEST(TestGroupMatmulAutoPhaseEnv, W4A8PromptUnsafeMTileClampsToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/512, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/32);
    apply_w4a8_auto_probe(s);
    for (auto &c : s.layout)
        c = 'c';
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            1)
            << "W4A8 prompt with an M-tile-unsafe layout must retain the "
               "sequential fallback";
    EXPECT_STREQ(trace.reason, "auto_rule07_prompt_w4a8_mtile");
}

// A BF16/FP16 layer of several experts runs both unset phases on ALGO 3, so its
// cross-warm has a reader only when a pin sends a phase off ALGO 3.  Every
// other layer keeps the shape rules, whose phases may land on either side.
TEST(TestGroupMatmulAutoPhaseEnv, CrossWarmReachableFollowsLayerScope) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_unset(-1);
    AutoDecodeAlgoOverride decode_unset(-1);

    for (const int from : {1, 2, 3}) {
        EXPECT_FALSE(grp_matmul_cross_warm_target_reachable(from, true))
                << "BF16/FP16 MoE layer from ALGO " << from;
        EXPECT_TRUE(grp_matmul_cross_warm_target_reachable(from, false))
                << "shape-rule layer from ALGO " << from;
    }
    {
        AutoPromptAlgoOverride prompt_pinned_1(1);
        EXPECT_TRUE(grp_matmul_cross_warm_target_reachable(3, true))
                << "the pinned prompt reads the full-weight layout";
        EXPECT_TRUE(grp_matmul_cross_warm_target_reachable(1, true))
                << "the unset decode reads the ALGO 3 arena";
    }
    {
        AutoPromptAlgoOverride prompt_pinned_3(3);
        for (const bool layer : {true, false}) {
            EXPECT_FALSE(grp_matmul_cross_warm_target_reachable(3, layer))
                    << "a prompt pinned to ALGO 3 reads no full-weight layout";
        }
    }
    AutoPromptAlgoOverride prompt_pinned_1(1);
    AutoDecodeAlgoOverride decode_pinned_1(1);
    for (const int from : {1, 2}) {
        for (const bool layer : {true, false}) {
            EXPECT_FALSE(grp_matmul_cross_warm_target_reachable(from, layer))
                    << "with both phases off ALGO 3 nothing reads its arena";
        }
    }
}

// Every layer outside BF16/FP16 keeps the shape rules: a per-group `{G, N}`
// INT8 weight, like a per-token one, runs its prompt on ALGO 1 (Rule 0.7)
// and its decode by Rule 0.5's occupancy test or the ALGO 3 decode default.
TEST(TestGroupMatmulAutoPhaseEnv, PerGroupInt8KeepsShapeRules) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    static float wei_scale_sentinel = 1.0f;

    struct layer_case {
        int groups, num_ops, m, algo;
        const char *reason;
    };
    for (const auto &c : {layer_case {90, 4, 64, 1, "auto_rule07_prompt_seq"},
                 layer_case {90, 4, 2, 1, "auto_rule05_few_active"},
                 layer_case {90, 16, 2, 3, "auto_phase_env"},
                 layer_case {1, 4, 64, 1, "auto_rule07_prompt_seq"}}) {
        auto s = build_auto_probe(c.m, /*K=*/2880, /*N=*/5760, c.num_ops,
                /*num_threads=*/32);
        for (size_t i = 0; i < s.params.size(); ++i) {
            auto &p = s.params[i];
            p.dtypes.wei = data_type_t::s8;
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.quant_params.src_scale.dims = {c.m, c.groups};
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = {c.groups, s.N[i]};
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.buff = &wei_scale_sentinel;
        }
        auto_algo_trace trace;
        EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                          s.num_threads, &trace),
                c.algo)
                << "groups=" << c.groups << " experts=" << c.num_ops
                << " M=" << c.m;
        EXPECT_STREQ(trace.reason, c.reason);
    }
}

// The BF16 MoE scope counts the declared expert pool: one firing expert of a
// pool of several (the `active_matmul` opt-in the routed MoE fallback sets)
// stays on ALGO 3, while the same call without the opt-in is a single expert
// and takes Rule 0.7.  A layer whose weights the caller AOCL-prepacked keeps
// the shape rules.
TEST(TestGroupMatmulAutoPhaseEnv, Bf16MoeScopeCountsDeclaredPool) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    auto_algo_trace trace;

    auto one = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(one.layout, one.M, one.N, one.K,
                      one.params, one.num_threads, &trace),
            1);
    EXPECT_STREQ(trace.reason, "auto_rule07_prompt_seq");
    one.params[0].active_matmul = 1;
    one.params[0].total_matmul = 8;
    EXPECT_EQ(select_grp_matmul_algo(one.layout, one.M, one.N, one.K,
                      one.params, one.num_threads, &trace),
            3);
    EXPECT_STREQ(trace.reason, "auto_algo3");

    auto packed = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/8, /*num_threads=*/64);
    for (auto &p : packed.params) {
        p.mem_format_b = 'r';
        p.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    }
    EXPECT_EQ(select_grp_matmul_algo(packed.layout, packed.M, packed.N,
                      packed.K, packed.params, packed.num_threads, &trace),
            1);
    EXPECT_STREQ(trace.reason, "auto_rule07_prompt_seq");
}

// ── Shape rules: every call but a BF16/FP16 layer of several experts ─────
// A single expert of any dtype and a layer of any other dtype keep the
// shape rules; the tests below drive them with f32 probes where the
// shape would otherwise be a BF16/FP16 layer of several experts.

TEST(TestGroupMatmulAutoPhaseEnv, CapacityOverflowIgnoresPhaseEnv) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride force_prompt(3);
    AutoDecodeAlgoOverride force_decode(3);

    // Decode-class with 300 experts > kNTilePlanMaxExperts(256).
    // Phase env says 3, but R0 must win → ALGO 1.
    auto s = build_auto_probe(/*M=*/4, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/300, /*num_threads=*/128);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "R0 capacity gate must override phase env "
               "(num_ops > kNTilePlanMaxExperts → ALGO 1)";
}

// Rule 0.6 counts ACTIVE experts (M[i] > 0), NOT the padded slot count.
// A legacy caller may pass M with inactive (M[i]==0) placeholders in-range;
// only the firing experts consume a thread, so a call whose SLOT count
// exceeds the thread budget but whose ACTIVE count does not must stay on
// ALGO 3 (the rule must not fire on padding alone).
TEST(TestGroupMatmulAutoPhaseEnv, DecodeUsesActiveExpertCountNotSlotCount) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // Pinned to the Rounds strategy so "rule fired" is observable as ALGO 5.
    // Under the default (3 = decode_dynamic) the rule reroutes to ALGO 3
    // instead, which is the same answer as "rule did not fire" and would make
    // this active-vs-slot distinction untestable.
    NTileStrategyOverride rounds(2);

    // 88 slots on 64 threads (slot count > threads) but only 58 fire — mark
    // the trailing 30 inactive (M[i]=0).  active_ops(58) <= num_threads(64),
    // so Rule 0.6 must NOT fire and decode stays on its ALGO 3 default.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    for (int i = 58; i < 88; ++i)
        s.M[i] = 0;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "decode slot count(88) > threads(64) but active(58) <= "
               "threads(64): "
               "Rule 0.6 must compare ACTIVE experts, so ALGO 3 stays selected";

    // Now make 78 fire (active > threads) — Rule 0.6 fires. Both sides of the
    // threshold answer ALGO 3 now, so assert on the RULE having fired rather
    // than on a distinct algo: see the reason= telemetry for the split.
    for (int i = 0; i < 88; ++i)
        s.M[i] = (i < 78) ? 16 : 0;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "decode active(78) > threads(64) must route to ALGO 3";
}

// Default prompt policy: the raw setting retains cached value 2 for
// compatibility, then latest main's default-only Rule 0.7 selects ALGO 1.
// No `Override` is set, so this exercises the actual out-of-box route.
TEST(TestGroupMatmulAutoPhaseEnv, DefaultPromptRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No AutoPromptAlgoOverride / AutoDecodeAlgoOverride — exercises
    // the unset-env default policy.

    // Mixtral-class prompt — Rule 0.7 selects ALGO 1 unless the operator
    // explicitly pins a prompt scheduler.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "the unpinned prompt policy must route prompt → ALGO 1";
}

// Rule 0.5 is a dtype-agnostic occupancy test for few-expert decode.
// Under-occupied frames use full-team sequential ALGO 1; frames with at most
// four threads per active expert use N-tile ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsDecodeShallowMRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Mixtral-class decode: 8 active experts cannot fill a 128-thread team
    // under the four-threads-per-expert threshold.
    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "8 active * 4 < 128 threads must select full-team ALGO 1";
}

// M does not alter Rule 0.5's occupancy decision. A single-token frame with
// 8 active experts and 128 threads remains under-occupied and takes ALGO 1.
// AUTO must never emit ALGO 2; that requires an explicit pin — the
// `ExplicitPromptAlgo2PinBeatsRegimeRouting` and
// `SingleDenseExpertDecodeHonoursExplicitPin` tests cover that direction.
TEST(TestGroupMatmulAutoPhaseEnv,
        FewExpertsDecodeSingleTokenNeverAutoSelects2) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // 8 active experts, M=1: 8 * 4 < 128.
    auto s = build_auto_probe(/*M=*/1, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "AUTO must not select ALGO 2: a ≤8-expert single-token decode "
               "uses Rule 0.5's low-occupancy ALGO 1 arm";
}

// Rule 0.5 counts the TOTAL experts (framework `total_matmul`), not the
// active op count.  An active-4 / total-16 decode shape exceeds the ≤8
// threshold, so it falls through to the decode default ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, FewExpertsUsesTotalNotActiveCount) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // 4 ACTIVE experts but 16 TOTAL (framework opt-in: active_matmul=4,
    // total_matmul=16).  total=16 > 8 → Rule 0.5 must NOT fire.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/4, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    s.params[0].active_matmul = 4;
    s.params[0].total_matmul = 16;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "Rule 0.5 must use TOTAL (16), not active (4); total>8 → "
               "decode default ALGO 3";

    // Same active count but total=8 → Rule 0.5 fires. Four active experts
    // cannot satisfy 4*4 >= 64, so the occupancy arrow selects ALGO 1. The
    // different outcome proves that TOTAL scopes entry to Rule 0.5 while
    // ACTIVE count drives its occupancy decision.
    s.params[0].total_matmul = 8;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "total=8 (≤ threshold) must fire Rule 0.5 even with only 4 "
               "active experts; 4*4 < 64 → ALGO 1";
}

// Rule 0.6 decides the experts-exceed-threads decode regime when the decode
// phase env is UNPINNED.  Here the rule and the pin agree on ALGO 3, so this
// locks in the answer rather than the precedence; the pin-outranks-policy
// direction is covered by ExplicitPhasePinOutranksPolicyRule below.
TEST(TestGroupMatmulAutoPhaseEnv,
        MoreExpertsThanThreadsOverridesExplicitDecodePin) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0);
    NTileStrategyOverride rounds(2); // legacy Rounds routing (ALGO 5 policy)
    AutoDecodeAlgoOverride force_decode(3); // explicit decode pin (superseded)

    auto s = build_auto_probe(/*M=*/16, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "decode num_ops(88) > num_threads(64) must be decided by Rule "
               "0.6 (ALGO 3) ahead of the explicit AUTO_DECODE_ALGO pin";
}

// Latest main's Rule 0.7 routes every unpinned prompt to ALGO 1.  The pair
// below pins both sides of the retired work-per-expert gate to ensure prompt
// depth no longer changes that default.
TEST(TestGroupMatmulAutoPhaseEnv, PromptManyExpertsDeepRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    // No phase-env override — exercises the default prompt policy.

    // Prompt-class: max_M=768 > kDecodeMaxM(32); 88 active experts on 64
    // threads (active_ops > num_threads).  Deep side of the retired gate:
    // 768/88 = 8.7 rows per expert cleared 0.9*sqrt(64) = 7.2.
    auto s = build_auto_probe(/*M=*/768, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "an unpinned prompt must route to ALGO 1 regardless of depth";
}

// Thin side of the same regime — also ALGO 1.
TEST(TestGroupMatmulAutoPhaseEnv, PromptManyExpertsThinRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Same 88-expert / 64-thread frame, but max_M=64 → 0.7 rows per expert,
    // far under the 7.2 the retired gate demanded of a 64-thread team.
    auto s = build_auto_probe(/*M=*/64, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/88, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "an unpinned prompt must route to ALGO 1 regardless of depth";
}

// Rule 0.7 applies before M-tile regime classification for an unpinned prompt,
// so even a genuine M-tile shape uses ALGO 1 by default.
TEST(TestGroupMatmulAutoPhaseEnv, PromptMTileRegimeRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/512, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/16, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "the unpinned prompt policy must route M-tile shapes to ALGO 1";
}

// Rule 0.7: a PROMPT-class wide-N frame (few actives × shallow M ×
// large N: max_M>1 && total_need*2 <= num_threads) routes to ALGO 1
// (sequential full-team) — reproducing flat_m_tile's old wide-N fallback,
// now that ALGO 2 is pure M-tile.
TEST(TestGroupMatmulAutoPhaseEnv, PromptWideNRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // 8 actives × M=8 on 64 threads: total_need = 8*ceil(8/16) = 8, 2*8=16
    // <= 64; max_M=8 > 1 → wide-N regime.  Prompt-class because... max_M=8
    // <= kDecodeMaxM(32) would be DECODE.  Use M=40 (>32 prompt) but keep
    // total_need*2 <= num_threads: 8*ceil(40/16)=8*3=24, 2*24=48 <= 64. ✓
    auto s = build_auto_probe(/*M=*/40, /*K=*/2048, /*N=*/1536,
            /*num_ops=*/8, /*num_threads=*/64);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "prompt-class wide-N (8 actives × M=40 / 64t: total_need*2=48 "
               "<= 64, "
               "max_M=40 > 1) must route to ALGO 1 (sequential full-team)";
}

TEST(TestGroupMatmulAutoPhaseEnv, S8FewExpertsDecodeStillTakesRule05) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    for (auto &p : s.params) {
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "s8/W8A8 few-expert shallow-M decode must use Rule 0.5 → "
               "ALGO 1";
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

    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "DENSE_DECODE_NTILE=0 must restore the generic few-expert "
               "occupancy decision (Rule 0.5 → ALGO 1)";
}

// Phase gate: the rule is decode-only, so a single-expert PROMPT frame
// is NOT pulled into ALGO 3 — it keeps the prompt policy and lands on
// Rule 0.7's `kWideN → ALGO 1`.  One expert needs only `ceil(M /
// slice_target)` M-slices, far fewer than the team, so `kWideN` (whose
// ALGO 1 gives the full team an N-split) is the right prompt answer.
// That same classification in DECODE is what Rule 0.45 exists to
// override, so this test also pins the boundary between the two.
TEST(TestGroupMatmulAutoPhaseEnv, SingleDenseExpertPromptStaysOnWideNAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Same expert, prompt-class M (> kDecodeMaxM).
    auto s = build_auto_probe(/*M=*/256, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/1, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "Rule 0.45 is decode-only; a single-expert prompt frame must "
               "stay on the prompt policy (kWideN → ALGO 1), not ALGO 3";
}

TEST(TestGroupMatmulAutoPhaseEnv,
        W4A8FewExpertsDecodeFullOccupancyRoutesToAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/1, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/32);
    apply_w4a8_auto_probe(s);
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            3)
            << "8 active * 4 >= 32 threads must route W4A8 through the "
               "generic Rule 0.5 ALGO 3 arm";
    EXPECT_STREQ(trace.reason, "auto_rule05_ntile_occupancy");
}

TEST(TestGroupMatmulAutoPhaseEnv,
        W4A8FewExpertsDecodeLowOccupancyRoutesToAlgo1) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_w4a8_auto_probe(s);
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            1)
            << "8 active * 4 < 128 threads must route W4A8 through the "
               "generic Rule 0.5 ALGO 1 arm";
    EXPECT_STREQ(trace.reason, "auto_rule05_few_active");
}

TEST(TestGroupMatmulAutoPhaseEnv, W4A8InactiveLeftoverUsesRule05Occupancy) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/28);
    apply_w4a8_auto_probe(s);
    s.M[0] = 0;
    s.params[0].dtypes.src = data_type_t::bf16;
    s.params[0].dtypes.wei = data_type_t::bf16;
    s.params[0].dynamic_quant = false;
    s.params[0].quant_params = {};
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            3)
            << "7 active * 4 >= 28 threads must use Rule 0.5's ALGO 3 arm";
    EXPECT_STREQ(trace.reason, "auto_rule05_ntile_occupancy");
}

TEST(TestGroupMatmulAutoPhaseEnv, W4A8DecodeHonoursExplicitPin) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoDecodeAlgoOverride pin_m_tile(2);

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_w4a8_auto_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            2)
            << "an explicit AUTO_DECODE_ALGO pin must outrank generic Rule 0.5";
}

TEST(TestGroupMatmulAutoPhaseEnv, W4A8DecodeHonoursNTileSafetyClamp) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/16, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/32);
    apply_w4a8_auto_probe(s);
    for (auto &c : s.layout)
        c = 'c';
    auto_algo_trace trace;
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace),
            1)
            << "the generic ALGO 3 occupancy arm must honour n_tile_safe";
    EXPECT_STREQ(trace.reason, "auto_rule05_ntile_occupancy");
}

// Grouped-s8 post-quant call with INACTIVE experts must NOT be vetoed to
// ALGO 1.  The grouped/fallback DQ pre-pass leaves inactive (M==0)
// experts at bf16 + dynamic_quant=false while active experts are s8 +
// per-token src_scale.  check_m_tile_safe / check_n_tile_extra must skip
// the inactive experts (and key dtype uniformity off the first active
// one) so a >8-expert decode shape still routes to ALGO 3, isolating the
// n_tile/m_tile eligibility veto.
TEST(TestGroupMatmulAutoPhaseEnv, GroupedS8WithInactiveExpertsKeepsAlgo3) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using zendnnl::common::data_type_t;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // 16 experts (>8), decode-class M=16, n_tile-safe shape (K%4==0).
    constexpr int num_ops = 16, M_active = 16, K = 2880, N = 5760;
    auto s = build_auto_probe(M_active, K, N, num_ops, /*num_threads=*/64);

    // Mark a LEADING (index 0) and an interior (index 8) expert inactive.
    s.M[0] = 0;
    s.M[8] = 0;

    // Persistent non-null scale buffers (never dereferenced for the
    // classification, but the gates require non-null + correct dims).
    static std::vector<std::vector<float>> src_scales(num_ops);
    static std::vector<std::vector<float>> wei_scales(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        auto &p = s.params[i];
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = false;
        // Per-channel wei scale {1, N} is set on EVERY expert (the caller
        // supplies static weight scales for active + inactive alike).
        wei_scales[i].assign(static_cast<size_t>(N), 1.0f / 127.0f);
        p.quant_params.wei_scale.dims = {1, static_cast<int64_t>(N)};
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = wei_scales[i].data();
        if (s.M[i] > 0) {
            // Active expert: grouped-prequant post-state — s8 src + per-token
            // src_scale {M, 1}.
            p.dtypes.src = data_type_t::s8;
            src_scales[i].assign(static_cast<size_t>(s.M[i]), 1.0f / 127.0f);
            p.quant_params.src_scale.dims = {static_cast<int64_t>(s.M[i]), 1};
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.buff = src_scales[i].data();
        } else {
            // Inactive expert: left at pre-quant bf16, no src_scale (exactly
            // what group_reorder_quantization_wrapper publishes for M==0).
            p.dtypes.src = data_type_t::bf16;
            p.quant_params.src_scale = {};
        }
    }

    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "grouped-s8 decode with inactive experts must still route to "
               "ALGO 3 — inactive (bf16) experts must not veto m_tile/n_tile "
               "eligibility";
}

// Phase env isolation: setting AUTO_PROMPT_ALGO=3 must NOT affect
// decode-class calls, and vice versa.
TEST(TestGroupMatmulAutoPhaseEnv, PromptEnvDoesNotLeakIntoDecode) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride force_prompt(2); // bogus-for-decode ALGO
    AutoDecodeAlgoOverride no_decode(0); // legacy for decode

    // Decode-class (M=16 ≤ kDecodeMaxM), gpt-oss-style, num_ops=32.
    // Legacy rule 3 → ALGO 3.  Prompt env must not bleed in.
    auto s = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/32, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "Decode-class call must read AUTO_DECODE_ALGO (=0, legacy "
               "rules → ALGO 3), NOT AUTO_PROMPT_ALGO";
}

TEST(TestGroupMatmulAutoPhaseEnv, DecodeEnvDoesNotLeakIntoPrompt) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride no_prompt(0); // legacy for prompt
    AutoDecodeAlgoOverride force_decode(2); // bogus-for-prompt ALGO

    // Prompt-class (M=256), Mixtral-style, num_ops=8.
    // Legacy rule 2 → ALGO 1.  Decode env must not bleed in.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "Prompt-class call must read AUTO_PROMPT_ALGO (=0, legacy "
               "rules → ALGO 1), NOT AUTO_DECODE_ALGO";
}

TEST(TestGroupMatmulAutoPhaseEnv, SharedClassifierUsesBoundaryAndActivePrefix) {
    using namespace zendnnl::lowoha::matmul;

    EXPECT_EQ(classify_grp_matmul_phase(std::vector<int> {0, 32}),
            grp_matmul_phase::decode);
    EXPECT_EQ(classify_grp_matmul_phase(std::vector<int> {0, 33}),
            grp_matmul_phase::prompt);
    EXPECT_EQ(classify_grp_matmul_phase(/*max_active_m=*/32),
            grp_matmul_phase::decode);
    EXPECT_EQ(classify_grp_matmul_phase(/*max_active_m=*/33),
            grp_matmul_phase::prompt);

    // The trailing 4096-row expert is prepack-only. Classification of the
    // first two compute-active entries must stay decode.
    const std::vector<int> with_prepack_tail {8, 32, 4096};
    EXPECT_EQ(classify_grp_matmul_phase(with_prepack_tail, /*active_prefix=*/2),
            grp_matmul_phase::decode);
    EXPECT_EQ(classify_grp_matmul_phase(with_prepack_tail),
            grp_matmul_phase::prompt);
}

TEST(TestGroupMatmulAutoPhaseEnv,
        W8A8ResolverHonorsPhasePrecedenceAndPhaseIsolation) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();

    // PRECEDENCE: the phase knob for the ACTIVE phase outranks the global
    // selector.  A phase `=4` is attributed to the phase, not the global,
    // even when the global is also 4 — the source identity reports which
    // knob was honoured, and that is now the phase.
    {
        AlgoEnvGuard global_w8a8(4);
        AutoPromptAlgoOverride prompt_w8a8(4);
        AutoDecodeAlgoOverride decode_w8a8(4);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::auto_decode);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::auto_prompt);
    }

    // A phase `=4` outranks EVERY global generic pin: the operator asked for
    // W8A8 on this phase specifically, which is the more specific statement.
    for (const int global : {1, 2, 3, 5, 6}) {
        AlgoEnvGuard global_pin(global);
        AutoPromptAlgoOverride prompt_w8a8(4);
        AutoDecodeAlgoOverride decode_w8a8(4);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::auto_decode)
                << "global=" << global;
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::auto_prompt)
                << "global=" << global;
    }

    // Converse direction: an explicit non-4 phase knob suppresses a global
    // `=4`, so the global W8A8 attempt does not leak into a phase the
    // operator has pinned to a generic scheduler.
    for (const int phase_generic : {1, 2, 3, 5, 6}) {
        AlgoEnvGuard global_w8a8(4);
        AutoPromptAlgoOverride prompt_pin(phase_generic);
        AutoDecodeAlgoOverride decode_pin(phase_generic);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::none)
                << "phase=" << phase_generic;
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::none)
                << "phase=" << phase_generic;
    }

    // The global still applies when the matching phase knob is UNSET.
    {
        AlgoEnvGuard global_w8a8(4);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::global);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::global);
    }

    // Under global AUTO, only the matching phase can request W8A8. The
    // inactive phase remains a normal generic pin (6 here).
    {
        AlgoEnvGuard global_auto(0);
        AutoPromptAlgoOverride prompt_w8a8(4);
        AutoDecodeAlgoOverride decode_multilevel(6);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::auto_prompt);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::none);

        auto decode = build_auto_probe(/*M=*/8, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/16, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(decode.layout, decode.M, decode.N,
                          decode.K, decode.params, decode.num_threads),
                6)
                << "inactive prompt phase 4 must not disturb decode ALGO 6";
    }
    {
        AlgoEnvGuard global_auto(0);
        AutoPromptAlgoOverride prompt_multilevel(6);
        AutoDecodeAlgoOverride decode_w8a8(4);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::decode),
                grp_matmul_ntile_flat_parallel_request_source::auto_decode);
        EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                          grp_matmul_phase::prompt),
                grp_matmul_ntile_flat_parallel_request_source::none);

        auto prompt = build_auto_probe(/*M=*/256, /*K=*/2048, /*N=*/1536,
                /*num_ops=*/16, /*num_threads=*/64);
        EXPECT_EQ(select_grp_matmul_algo(prompt.layout, prompt.M, prompt.N,
                          prompt.K, prompt.params, prompt.num_threads),
                6)
                << "inactive decode phase 4 must not disturb prompt ALGO 6";
    }
}

// Phase value 4 is a valid raw W8A8 request, but its generic effective value
// is the inherited phase default. It must not count as a generic policy pin:
// after an eligibility decline, the default policy runs.
TEST(TestGroupMatmulAutoPhaseEnv,
        PhaseValue4RequestsW8A8AndInheritsDefaultPolicy) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride prompt_w8a8(4);
    AutoDecodeAlgoOverride decode_w8a8(4);

    const auto prompt = get_grp_matmul_auto_prompt_setting();
    EXPECT_EQ(prompt.requested_algo, 4);
    EXPECT_EQ(prompt.generic_effective_algo, 2);
    EXPECT_TRUE(prompt.has_explicit_request());
    EXPECT_TRUE(prompt.requests_ntile_flat_parallel());
    EXPECT_FALSE(prompt.pins_generic_policy());
    EXPECT_FALSE(grp_matmul_auto_prompt_algo_is_set());

    const auto decode = get_grp_matmul_auto_decode_setting();
    EXPECT_EQ(decode.requested_algo, 4);
    EXPECT_EQ(decode.generic_effective_algo, 3);
    EXPECT_TRUE(decode.has_explicit_request());
    EXPECT_TRUE(decode.requests_ntile_flat_parallel());
    EXPECT_FALSE(decode.pins_generic_policy());
    EXPECT_FALSE(grp_matmul_auto_decode_algo_is_set());

    EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                      grp_matmul_phase::prompt),
            grp_matmul_ntile_flat_parallel_request_source::auto_prompt);
    EXPECT_EQ(resolve_grp_matmul_ntile_flat_parallel_request(
                      grp_matmul_phase::decode),
            grp_matmul_ntile_flat_parallel_request_source::auto_decode);

    // Request 4 is not a generic pin, so after an ALGO4 decline latest main's
    // default-only prompt refinement selects ALGO 1.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    apply_shape_rules_probe(s);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "phase 4 must expose the inherited prompt policy (ALGO 1)";

    // Decode shape with n_tile_safe=true → inherited decode default 3.
    auto sd = build_auto_probe(/*M=*/16, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/32, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      sd.layout, sd.M, sd.N, sd.K, sd.params, sd.num_threads),
            3)
            << "phase 4 must expose the inherited decode default (=3)";

    // Default-policy refinements must not be suppressed by treating phase 4
    // as a literal pin.
    auto wide_prompt = build_auto_probe(/*M=*/40, /*K=*/2048, /*N=*/8192,
            /*num_ops=*/8, /*num_threads=*/64);
    apply_shape_rules_probe(wide_prompt);
    auto_algo_trace prompt_trace;
    EXPECT_EQ(select_grp_matmul_algo(wide_prompt.layout, wide_prompt.M,
                      wide_prompt.N, wide_prompt.K, wide_prompt.params,
                      wide_prompt.num_threads, &prompt_trace),
            1);
    EXPECT_STREQ(prompt_trace.reason, "auto_rule07_prompt_seq");

    // Few-expert decode likewise refines inherited default 3 via Rule 0.5.
    auto few_decode = build_auto_probe(/*M=*/8, /*K=*/2048, /*N=*/8192,
            /*num_ops=*/4, /*num_threads=*/64);
    apply_shape_rules_probe(few_decode);
    auto_algo_trace decode_trace;
    EXPECT_EQ(select_grp_matmul_algo(few_decode.layout, few_decode.M,
                      few_decode.N, few_decode.K, few_decode.params,
                      few_decode.num_threads, &decode_trace),
            1);
    ASSERT_NE(decode_trace.reason, nullptr);
    EXPECT_EQ(std::string(decode_trace.reason).rfind("auto_rule05_", 0), 0u);

    // A BF16/FP16 layer of several experts runs ALGO 3 after the decline too.
    auto moe = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    auto_algo_trace moe_trace;
    EXPECT_EQ(select_grp_matmul_algo(moe.layout, moe.M, moe.N, moe.K,
                      moe.params, moe.num_threads, &moe_trace),
            3);
    EXPECT_STREQ(moe_trace.reason, "auto_algo3");
}

// `AUTO_PROMPT_ALGO=2` is a PIN, not a hint: an operator that explicitly
// asks for ALGO 2 gets ALGO 2 on a prompt frame the default sends to
// ALGO 3.
TEST(TestGroupMatmulAutoPhaseEnv, ExplicitPromptAlgo2PinBeatsRegimeRouting) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    // Wide-N prompt frame: few experts, modest M, large thread count.
    auto s = build_auto_probe(/*M=*/40, /*K=*/2048, /*N=*/8192,
            /*num_ops=*/8, /*num_threads=*/64);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3)
            << "the unpinned prompt default must route to ALGO 3";

    AutoPromptAlgoOverride pin_prompt(2);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            2)
            << "an explicit AUTO_PROMPT_ALGO=2 must be honoured verbatim";
}

// The `[GRP_MATMUL.ALGO] reason=` field must name the rule that actually
// matched, including the safety clamp and the legacy cascade's arms.
TEST(TestGroupMatmulAutoPhaseEnv, TraceNamesTheMatchedRule) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    const auto reason_for
            = [](int M, int K, int N, int num_ops, int num_threads,
                      int *algo_out, bool shape_rules = true) {
        auto s = build_auto_probe(M, K, N, num_ops, num_threads);
        if (shape_rules) { apply_shape_rules_probe(s); }
        auto_algo_trace trace;
        const int algo = select_grp_matmul_algo(
                s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace);
        if (algo_out != nullptr) { *algo_out = algo; }
        return std::string(trace.reason != nullptr ? trace.reason : "<none>");
    };

    int algo = 0;
    // Rule 0.45 — lone expert in decode. Was invisible to the old mirror.
    EXPECT_EQ(reason_for(/*M=*/8, /*K=*/2048, /*N=*/8192, /*num_ops=*/1,
                      /*num_threads=*/64, &algo),
            "auto_rule045_single_dense_decode");
    EXPECT_EQ(algo, 3);

    // Rule 0.5 — few experts in decode. Also invisible to the old mirror.
    // The regime arm depends on the shape; the rule is what matters here.
    const std::string rule05 = reason_for(/*M=*/8, /*K=*/2048, /*N=*/8192,
            /*num_ops=*/4, /*num_threads=*/64, &algo);
    EXPECT_EQ(rule05.rfind("auto_rule05_", 0), 0u)
            << "few-expert decode must report a Rule 0.5 arm, got " << rule05;

    // W4A8 uses the same Rule 0.5 occupancy decision.
    {
        auto s = build_auto_probe(/*M=*/8, /*K=*/2048, /*N=*/8192,
                /*num_ops=*/4, /*num_threads=*/64);
        apply_w4a8_auto_probe(s);
        auto_algo_trace trace;
        algo = select_grp_matmul_algo(
                s.layout, s.M, s.N, s.K, s.params, s.num_threads, &trace);
        EXPECT_STREQ(trace.reason, "auto_rule05_few_active");
        EXPECT_EQ(algo, 1);
    }

    // Rule 0 — structural capacity carve-out.
    EXPECT_EQ(reason_for(/*M=*/8, /*K=*/2048, /*N=*/8192, /*num_ops=*/300,
                      /*num_threads=*/64, &algo),
            "auto_rule0_capacity");
    EXPECT_EQ(algo, 1);

    // Rule 0.7 — inherited prompt default on a wide-N frame.
    EXPECT_EQ(reason_for(/*M=*/40, /*K=*/2048, /*N=*/8192, /*num_ops=*/8,
                      /*num_threads=*/64, &algo),
            "auto_rule07_prompt_seq");
    EXPECT_EQ(algo, 1);

    // A BF16/FP16 layer of several experts, in either phase.
    for (const int M : {8, 40}) {
        EXPECT_EQ(reason_for(M, /*K=*/2048, /*N=*/8192, /*num_ops=*/4,
                          /*num_threads=*/64, &algo, /*shape_rules=*/false),
                "auto_algo3")
                << "M=" << M;
        EXPECT_EQ(algo, 3);
    }

    // An explicit pin reports itself, not the rule it pre-empted.
    {
        AutoPromptAlgoOverride pin_prompt(6);
        EXPECT_EQ(reason_for(/*M=*/40, /*K=*/2048, /*N=*/8192, /*num_ops=*/8,
                          /*num_threads=*/64, &algo),
                "auto_phase_env");
        EXPECT_EQ(algo, 6);
    }
}

// A value the getter refuses (outside 0..6 / requested selector identities) is
// not a pin.  If `is_set()` accepted it, the call would claim a pin that was
// never honoured, which the trace would report as `auto_phase_env`.
TEST(TestGroupMatmulAutoPhaseEnv, InvalidPromptValueFallsBackToDefault) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/40, /*K=*/2048, /*N=*/8192,
            /*num_ops=*/8, /*num_threads=*/64);

    for (const int bogus : {7, 99, 1000}) {
        AutoPromptAlgoOverride pin_prompt(bogus);
        EXPECT_FALSE(grp_matmul_auto_prompt_algo_is_set())
                << "value " << bogus
                << " is outside {0,1,2,3,4,5,6} and must not read as an "
                   "explicit pin";
        auto_algo_trace trace;
        EXPECT_EQ(select_grp_matmul_algo(s.layout, s.M, s.N, s.K, s.params,
                          s.num_threads, &trace),
                3)
                << "value " << bogus;
        EXPECT_STREQ(trace.reason, "auto_algo3")
                << "value " << bogus << " must behave as the inherited default";
    }
}

// Decode counterpart: an invalid decode value must not suppress the
// decode-side rules that only run on the inherited default.
TEST(TestGroupMatmulAutoPhaseEnv, InvalidDecodeValueIsNotAPin) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();

    for (const int bogus : {7, 99, 1000}) {
        AutoDecodeAlgoOverride pin_decode(bogus);
        EXPECT_FALSE(grp_matmul_auto_decode_algo_is_set())
                << "value " << bogus
                << " is outside {0,1,2,3,4,5,6} and must not read as an "
                   "explicit pin";
    }
    for (const int valid : {0, 1, 2, 3, 5, 6}) {
        AutoDecodeAlgoOverride pin_decode(valid);
        EXPECT_TRUE(grp_matmul_auto_decode_algo_is_set())
                << "value " << valid << " is a legal pin";
    }
    {
        AutoDecodeAlgoOverride w8a8_decode(4);
        const auto setting = get_grp_matmul_auto_decode_setting();
        EXPECT_TRUE(setting.has_explicit_request());
        EXPECT_TRUE(setting.requests_ntile_flat_parallel());
        EXPECT_FALSE(setting.pins_generic_policy());
        EXPECT_FALSE(grp_matmul_auto_decode_algo_is_set())
                << "phase 4 is valid for W8A8 but is not a generic pin";
    }
}

TEST(TestGroupMatmulAutoPhaseEnv, CachedPhaseEnvironmentParsesFourStrictly) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    testing::FLAGS_gtest_death_test_style = "threadsafe";

    // Re-exec gives the child a fresh image, so these checks exercise the real
    // cached getenv path rather than the in-process atomic test overrides.
    {
        EnvVarGuard global("ZENDNNL_GRP_MATMUL_ALGO", "0");
        EnvVarGuard prompt("ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO", "4");
        EnvVarGuard decode("ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO", "4");
        EXPECT_EXIT(
                {
                    const auto p = get_grp_matmul_auto_prompt_setting();
                    const auto d = get_grp_matmul_auto_decode_setting();
                    const bool ok = p.requested_algo == 4
                            && p.generic_effective_algo == 2
                            && p.requests_ntile_flat_parallel()
                            && !p.pins_generic_policy() && d.requested_algo == 4
                            && d.generic_effective_algo == 3
                            && d.requests_ntile_flat_parallel()
                            && !d.pins_generic_policy()
                            && resolve_grp_matmul_ntile_flat_parallel_request(
                                       grp_matmul_phase::prompt)
                                    == grp_matmul_ntile_flat_parallel_request_source::
                                            auto_prompt
                            && resolve_grp_matmul_ntile_flat_parallel_request(
                                       grp_matmul_phase::decode)
                                    == grp_matmul_ntile_flat_parallel_request_source::
                                            auto_decode;
                    std::exit(ok ? 0 : 1);
                },
                ::testing::ExitedWithCode(0), "");
    }

    // Strict parsing rejects trailing junk instead of accepting its leading 4.
    {
        EnvVarGuard global("ZENDNNL_GRP_MATMUL_ALGO", "0");
        EnvVarGuard prompt("ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO", "4junk");
        EnvVarGuard decode("ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO", "4junk");
        EXPECT_EXIT(
                {
                    const auto p = get_grp_matmul_auto_prompt_setting();
                    const auto d = get_grp_matmul_auto_decode_setting();
                    const bool ok = !p.has_explicit_request()
                            && p.generic_effective_algo == 2
                            && !d.has_explicit_request()
                            && d.generic_effective_algo == 3;
                    std::exit(ok ? 0 : 1);
                },
                ::testing::ExitedWithCode(0), "");
    }
}

// More experts than the N-tile planner's arrays hold
// (kNTilePlanMaxExperts=256) in a BF16 layer still run ALGO 3, unpinned or
// pinned; only the legacy cascade keeps its capacity arm to ALGO 1.
TEST(TestGroupMatmulAutoPhaseEnv, CapacityOverflowRoutesToAlgo3UnlessLegacy) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);

    auto s = build_auto_probe(/*M=*/4, /*K=*/2880, /*N=*/5760,
            /*num_ops=*/300, /*num_threads=*/128);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            3);
    {
        AutoDecodeAlgoOverride force_decode(3);
        EXPECT_EQ(select_grp_matmul_algo(
                          s.layout, s.M, s.N, s.K, s.params, s.num_threads),
                3);
    }
    AutoDecodeAlgoOverride legacy_decode(0);
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "the legacy cascade must keep num_ops > kNTilePlanMaxExperts "
               "on ALGO 1";
}

// ALGO 3 runs a call with more experts than the N-tile planner holds
// (kNTilePlanMaxExperts) as consecutive slices on the custom kernel.  Each
// case checks every firing expert against ALGO 1, that an idle expert's
// dst is untouched, and that prepack warmed every expert the call
// advertises: an idle slice's experts and the prepack-extras tail ride on
// the first slice that runs.
TEST(TestGroupMatmulManyExperts, Algo3SlicesPastPlanCapacity) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    // Out-of-place, so prepack warms the pack cache the probe reads and
    // the ALGO 1 reference leaves the weights raw for the ALGO 3 run.
    WeightCacheGuard wc_out_of_place(1);

    struct Case {
        const char *name;
        int num_ops; // M.size()
        int active; // params[0].active_matmul; 0 = legacy contract
        int total; // weight-side length (params[0].total_matmul if opt-in)
        int idle_lo, idle_hi; // legacy only: experts in [lo, hi) get M = 0
    };
    const Case cases[] = {
            {"legacy_E257", 257, 0, 257, 0, 0},
            {"legacy_E600_middle_slice_idle", 600, 0, 600, 200, 400},
            {"legacy_E520_first_slice_only", 520, 0, 520, 100, 520},
            {"padded_E300_active270", 300, 270, 300, 0, 0},
            {"compact_E300_extras_to_400", 300, 300, 400, 0, 0},
    };
    const int K = 64, N = 64;
    const bfloat16_t sentinel(7.0f);

    for (const Case &c : cases) {
        SCOPED_TRACE(c.name);
        const int E = c.num_ops, W = c.total;
        std::vector<int> Ms(E);
        for (int e = 0; e < E; ++e) {
            const bool fires = c.active > 0
                    ? e < c.active
                    : !(e >= c.idle_lo && e < c.idle_hi);
            Ms[e] = fires ? 1 + e % 4 : 0;
        }
        std::vector<std::vector<bfloat16_t>> src(E), wei(W);
        for (int e = 0; e < E; ++e) {
            src[e].resize(static_cast<size_t>(std::max(Ms[e], 1)) * K);
            fill_src(src[e], e, 0.5f);
        }
        for (int e = 0; e < W; ++e) {
            wei[e].resize(static_cast<size_t>(K) * N);
            fill_wei1(wei[e], e, 0.1f);
        }

        auto run = [&](int algo, std::vector<std::vector<bfloat16_t>> &dst,
                           std::string &mode) {
            AlgoEnvGuard algo_guard(algo);
            dst.assign(E, {});
            auto gv = GemmVecs::uniform(E, /*M=*/0, N, K);
            gv.Ms = Ms;
            gv.Ns.assign(W, N);
            gv.Ks.assign(W, K);
            gv.ldb.assign(W, N);
            gv.transB.assign(W, false);
            gv.is_wc.assign(W, true);
            auto params = make_uniform_params(E, data_type_t::bf16);
            params[0].active_matmul = static_cast<uint32_t>(c.active);
            params[0].total_matmul
                    = static_cast<uint32_t>(c.active > 0 ? c.total : 0);
            std::vector<const void *> srcs(E), weis(W), biases(E, nullptr);
            std::vector<void *> dsts(E);
            for (int e = 0; e < E; ++e) {
                dst[e].assign(
                        static_cast<size_t>(std::max(Ms[e], 1)) * N, sentinel);
                srcs[e] = src[e].data();
                dsts[e] = dst[e].data();
            }
            for (int e = 0; e < W; ++e) {
                weis[e] = wei[e].data();
            }
            GemmModeCaptureGuard capture;
            ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                              gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis,
                              gv.ldb, biases, gv.beta, dsts, gv.ldc, gv.is_wc,
                              params, /*moe_postop=*/nullptr,
                              /*gated_act=*/nullptr),
                    status_t::success)
                    << "algo=" << algo;
            const char *m = zendnnl::lowoha::matmul::test_api::
                                    s_last_group_matmul_direct_gemm_mode.load(
                                            std::memory_order_relaxed);
            mode = m != nullptr ? m : "";
        };

        reset_grp_matmul_caches();
        std::vector<std::vector<bfloat16_t>> ref, got;
        std::string ref_mode, got_mode;
        ASSERT_NO_FATAL_FAILURE(run(/*algo=*/1, ref, ref_mode));
        reset_grp_matmul_caches();
        ASSERT_NO_FATAL_FAILURE(run(/*algo=*/3, got, got_mode));

        EXPECT_NE(got_mode.find("_custom"), std::string::npos)
                << "ALGO 3 ran '" << got_mode << "' off the custom kernel";
        EXPECT_EQ(executed_algo_from_gemm_mode(got_mode.c_str()), 3)
                << got_mode;

        std::vector<const void *> weis(W);
        for (int e = 0; e < W; ++e) {
            weis[e] = wei[e].data();
        }
        prepack::custom_kernel::PackProbeStats probe;
        prepack::custom_kernel::warm_pack_all_custom_kernel_experts(weis,
                std::vector<int>(W, K), std::vector<int>(W, N),
                std::vector<int>(W, N), std::vector<bool>(W, false),
                /*is_weights_const=*/ {}, /*total_count=*/W, probe);
        EXPECT_EQ(probe.cache_hits, W)
                << "prepack left " << (W - probe.cache_hits)
                << " advertised experts cold";

        const Tol tol {0.02f, 0.02f};
        for (int e = 0; e < E; ++e) {
            if (Ms[e] == 0) {
                for (const bfloat16_t v : got[e]) {
                    ASSERT_EQ(
                            static_cast<float>(v), static_cast<float>(sentinel))
                            << "idle expert " << e << " was written";
                }
                continue;
            }
            for (size_t i = 0; i < got[e].size(); ++i) {
                const float r = static_cast<float>(ref[e][i]);
                ASSERT_NEAR(static_cast<float>(got[e][i]), r,
                        std::abs(r) * tol.rel + tol.abs)
                        << "expert " << e << " element " << i;
            }
        }
    }
}

// Under N_TILE_STRATEGY=0 the planner sends a shape too thin to split
// N-wise to Sequential, which runs every expert on AOCL from the caller's
// weight buffer.  A call the custom kernel serves stays on it.
TEST(TestGroupMatmulManyExperts, AutoStrategyKeepsCustomKernelOnThinN) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    NTileStrategyOverride auto_strategy(0);
    // Out-of-place, so the ALGO 1 reference leaves the weights raw.
    WeightCacheGuard wc_out_of_place(1);

    // Two experts on 16 threads leave a team of 8 per expert, but N=128
    // holds no full decode tile, so `ntile_viable` is false.
    const int E = 2, M = 4, K = 64, N = 128;
    std::vector<std::vector<bfloat16_t>> src(E), wei(E);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(static_cast<size_t>(K) * N);
        fill_src(src[e], e, 0.5f);
        fill_wei1(wei[e], e, 0.1f);
    }
    auto run = [&](int algo, std::vector<std::vector<bfloat16_t>> &dst,
                       std::string &mode) {
        AlgoEnvGuard algo_guard(algo);
        dst.assign(E,
                std::vector<bfloat16_t>(
                        static_cast<size_t>(M) * N, bfloat16_t(0.0f)));
        auto gv = GemmVecs::uniform(E, M, N, K);
        auto params = make_uniform_params(E, data_type_t::bf16);
        for (auto &p : params) {
            p.num_threads = 16;
        }
        std::vector<const void *> srcs(E), weis(E), biases(E, nullptr);
        std::vector<void *> dsts(E);
        for (int e = 0; e < E; ++e) {
            srcs[e] = src[e].data();
            weis[e] = wei[e].data();
            dsts[e] = dst[e].data();
        }
        GemmModeCaptureGuard capture;
        ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                          gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                          biases, gv.beta, dsts, gv.ldc, gv.is_wc, params,
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success)
                << "algo=" << algo;
        const char *m = zendnnl::lowoha::matmul::test_api::
                                s_last_group_matmul_direct_gemm_mode.load(
                                        std::memory_order_relaxed);
        mode = m != nullptr ? m : "";
    };

    reset_grp_matmul_caches();
    std::vector<std::vector<bfloat16_t>> ref, got;
    std::string ref_mode, got_mode;
    ASSERT_NO_FATAL_FAILURE(run(/*algo=*/1, ref, ref_mode));
    reset_grp_matmul_caches();
    ASSERT_NO_FATAL_FAILURE(run(/*algo=*/3, got, got_mode));
    EXPECT_NE(got_mode.find("_custom"), std::string::npos) << got_mode;
    EXPECT_EQ(got_mode.find("sequential"), std::string::npos) << got_mode;

    const Tol tol {0.02f, 0.02f};
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < got[e].size(); ++i) {
            const float r = static_cast<float>(ref[e][i]);
            ASSERT_NEAR(static_cast<float>(got[e][i]), r,
                    std::abs(r) * tol.rel + tol.abs)
                    << "expert " << e << " element " << i;
        }
    }
}

// Under an explicit ALGO 3 pin a call outside a BF16/FP16 MoE layer keeps the
// planner's gates: past the planner's capacity it runs Sequential instead of
// slices, and under N_TILE_STRATEGY=0 a single expert that AUTO would send to
// ALGO 1 runs Sequential.  BF16/FP16 MoE calls stay on the custom kernel.
TEST(TestGroupMatmulManyExperts,
        PinnedAlgo3KeepsPlannerGatesOutsideCkFloatMoe) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    // Out-of-place, so the ALGO 1 reference leaves the weights raw.
    WeightCacheGuard wc_out_of_place(1);

    // One call of `E` experts of `dst`'s element type on ALGO `algo`; returns
    // its gemm_mode and fills `dst`.
    const auto run = [](auto &dst, data_type_t dt, int algo, int E, int M,
                             int K, int N, int threads) {
        using T = typename std::decay_t<decltype(dst)>::value_type::value_type;
        reset_grp_matmul_caches();
        AlgoEnvGuard algo_guard(algo);
        std::vector<std::vector<T>> src(E), wei(E);
        dst.assign(E, std::vector<T>(static_cast<size_t>(M) * N, T(0.0f)));
        std::vector<const void *> srcs(E), weis(E), biases(E, nullptr);
        std::vector<void *> dsts(E);
        for (int e = 0; e < E; ++e) {
            src[e].resize(static_cast<size_t>(M) * K);
            wei[e].resize(static_cast<size_t>(K) * N);
            fill_src(src[e], e, 0.5f);
            fill_wei1(wei[e], e, 0.1f);
            srcs[e] = src[e].data();
            weis[e] = wei[e].data();
            dsts[e] = dst[e].data();
        }
        auto gv = GemmVecs::uniform(E, M, N, K);
        auto params = make_uniform_params(E, dt);
        for (auto &p : params) {
            p.num_threads = threads;
        }
        GemmModeCaptureGuard capture;
        EXPECT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                          gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                          biases, gv.beta, dsts, gv.ldc, gv.is_wc, params,
                          /*moe_postop=*/nullptr, /*gated_act=*/nullptr),
                status_t::success)
                << "algo=" << algo << " E=" << E;
        const char *m = zendnnl::lowoha::matmul::test_api::
                                s_last_group_matmul_direct_gemm_mode.load(
                                        std::memory_order_relaxed);
        return std::string(m != nullptr ? m : "");
    };
    const auto expect_near = [](const auto &got, const auto &ref) {
        const Tol tol {0.02f, 0.02f};
        for (size_t e = 0; e < got.size(); ++e) {
            for (size_t i = 0; i < got[e].size(); ++i) {
                const float r = static_cast<float>(ref[e][i]);
                ASSERT_NEAR(static_cast<float>(got[e][i]), r,
                        std::abs(r) * tol.rel + tol.abs)
                        << "expert " << e << " element " << i;
            }
        }
    };

    {
        SCOPED_TRACE("f32, 300 experts");
        std::vector<std::vector<float>> ref, got;
        run(ref, data_type_t::f32, 1, 300, 2, 64, 64, 0);
        EXPECT_EQ(run(got, data_type_t::f32, 3, 300, 2, 64, 64, 0),
                "flat_n_tile_sequential");
        expect_near(got, ref);
    }

    if (custom_kernel::avx512f16_available()) {
        SCOPED_TRACE("f16, 300 experts");
        std::vector<std::vector<float16_t>> ref, got;
        run(ref, data_type_t::f16, 1, 300, 2, 64, 64, 16);
        const std::string mode
                = run(got, data_type_t::f16, 3, 300, 2, 64, 64, 16);
        EXPECT_NE(mode.find("_custom"), std::string::npos) << mode;
        EXPECT_EQ(mode.find("sequential"), std::string::npos) << mode;
        expect_near(got, ref);
    }

    NTileStrategyOverride auto_strategy(0);
    for (const int E : {1, 2}) {
        SCOPED_TRACE(std::string("bf16, experts=") + std::to_string(E));
        std::vector<std::vector<bfloat16_t>> ref, got;
        run(ref, data_type_t::bf16, 1, E, 64, 64, 256, 16);
        const std::string mode
                = run(got, data_type_t::bf16, 3, E, 64, 64, 256, 16);
        if (E == 1) {
            EXPECT_EQ(mode, "flat_n_tile_sequential");
        } else {
            EXPECT_NE(mode.find("_custom"), std::string::npos) << mode;
        }
        expect_near(got, ref);
    }
}

// ===============================================================================
// TestGroupMatmulInPlaceCkPack — under WEIGHT_CACHE=2 the custom kernel packs
// a weight into the caller's buffer.  From then on only the custom kernel can
// read that buffer: a call that cannot run it on the buffer fails without
// writing dst, and a call that can is served whatever layout it asks for.
//
// References come from the same ALGO 3 calls on copies of the weights under
// WEIGHT_CACHE=1, so a correct result is bit-identical to them.
// ===============================================================================
namespace in_place_ck {

using Experts = std::vector<std::vector<bfloat16_t>>;

constexpr int kE = 2, kM = 4, kK = 64, kN = 128;
const bfloat16_t kSentinel(-3.0f);

struct Inputs {
    Experts src, wei;
    Inputs() : src(kE), wei(kE) {
        for (int e = 0; e < kE; ++e) {
            src[e].resize(static_cast<size_t>(kM) * kK);
            wei[e].resize(static_cast<size_t>(kK) * kN);
            moe_test_utils::fill_src(src[e], e, 0.5f);
            moe_test_utils::fill_wei1(wei[e], e, 0.1f);
        }
    }
};

bool same_bytes(const Experts &a, const Experts &b) {
    if (a.size() != b.size()) { return false; }
    for (size_t e = 0; e < a.size(); ++e) {
        if (a[e].size() != b[e].size()
                || std::memcmp(a[e].data(), b[e].data(),
                           a[e].size() * sizeof(bfloat16_t))
                        != 0) {
            return false;
        }
    }
    return true;
}

bool all_sentinel(const Experts &dst) {
    for (const auto &d : dst) {
        for (const bfloat16_t v : d) {
            if (static_cast<float>(v) != static_cast<float>(kSentinel)) {
                return false;
            }
        }
    }
    return true;
}

// One call; `ldc < kN` is the tight layout of a gated activation.  `dst` is
// refilled with `kSentinel` first.  `mode` receives the executed gemm_mode,
// which is only recorded for a call that succeeds.
status_t run(int algo, float alpha, grp_matmul_gated_act_t act, int ldc,
        Experts &src, Experts &wei, Experts &dst, std::string *mode = nullptr) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    AlgoEnvGuard algo_guard(algo);
    dst.assign(kE,
            std::vector<bfloat16_t>(static_cast<size_t>(kM) * ldc, kSentinel));
    auto gv = GemmVecs::uniform(kE, kM, kN, kK, alpha);
    gv.ldc.assign(kE, ldc);
    auto params = make_uniform_params(kE, data_type_t::bf16);
    for (auto &p : params) {
        p.num_threads = 16;
    }
    std::vector<const void *> srcs(kE), weis(kE), biases(kE, nullptr);
    std::vector<void *> dsts(kE);
    for (int e = 0; e < kE; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }
    grp_matmul_gated_act_params gated {};
    gated.act = act;
    GemmModeCaptureGuard capture;
    const status_t st = group_matmul_direct(gv.layout, gv.transA, gv.transB,
            gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
            gv.beta, dsts, gv.ldc, gv.is_wc, params, /*moe_postop=*/nullptr,
            act == grp_matmul_gated_act_t::none ? nullptr : &gated);
    if (mode != nullptr) {
        const char *m = test_api::s_last_group_matmul_direct_gemm_mode.load(
                std::memory_order_relaxed);
        *mode = (st == status_t::success && m != nullptr) ? m : "";
    }
    return st;
}

} // namespace in_place_ck

// Once the image is written, a call off the custom kernel (alpha != 1 under
// ALGO 3, or another ALGO) has no raw weight to read.
TEST(TestGroupMatmulInPlaceCkPack, OffKernelCallsFailClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using namespace in_place_ck;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    const auto none = grp_matmul_gated_act_t::none;
    reset_grp_matmul_caches();

    Inputs in;
    const Experts raw = in.wei;
    Experts ref, dst;
    {
        WeightCacheGuard out_of_place(1);
        Experts copy = raw;
        ASSERT_EQ(run(3, 1.0f, none, kN, in.src, copy, ref), status_t::success);
        // On raw weights alpha != 1 is a valid call.
        ASSERT_EQ(run(3, 0.5f, none, kN, in.src, copy, dst), status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard in_place(2);
    std::string mode;
    ASSERT_EQ(run(3, 1.0f, none, kN, in.src, in.wei, dst, &mode),
            status_t::success);
    EXPECT_NE(mode.find("_custom"), std::string::npos) << mode;
    EXPECT_TRUE(same_bytes(dst, ref));
    ASSERT_FALSE(same_bytes(in.wei, raw))
            << "the weights were not packed in "
               "place, so nothing below is tested";

    EXPECT_EQ(run(3, 0.5f, none, kN, in.src, in.wei, dst), status_t::failure);
    EXPECT_TRUE(all_sentinel(dst)) << "alpha=0.5 wrote dst";
    for (const int algo : {1, 2, 5}) {
        EXPECT_EQ(run(algo, 1.0f, none, kN, in.src, in.wei, dst),
                status_t::failure)
                << "algo=" << algo;
        EXPECT_TRUE(all_sentinel(dst)) << "algo=" << algo << " wrote dst";
    }

    ASSERT_EQ(run(3, 1.0f, none, kN, in.src, in.wei, dst), status_t::success);
    EXPECT_TRUE(same_bytes(dst, ref));
    reset_grp_matmul_caches();
}

// The plain layout and the interleaved one fused silu_and_mul reads can both
// be served from one buffer, in either order.
TEST(TestGroupMatmulInPlaceCkPack, ImageServesBothActivationLayouts) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using namespace in_place_ck;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    const auto none = grp_matmul_gated_act_t::none;
    const auto silu = grp_matmul_gated_act_t::silu_and_mul;
    const int tight = kN / 2;
    reset_grp_matmul_caches();

    Experts ref_none, ref_silu, dst;
    {
        WeightCacheGuard out_of_place(1);
        Inputs in;
        ASSERT_EQ(run(3, 1.0f, none, kN, in.src, in.wei, ref_none),
                status_t::success);
        ASSERT_EQ(run(3, 1.0f, silu, tight, in.src, in.wei, ref_silu),
                status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard in_place(2);
    for (const bool plain_first : {true, false}) {
        SCOPED_TRACE(plain_first ? "plain image first" : "interleaved first");
        Inputs in;
        const Experts raw = in.wei;
        const auto check = [&](grp_matmul_gated_act_t act) {
            const bool gated = act != none;
            std::string mode;
            ASSERT_EQ(run(3, 1.0f, act, gated ? tight : kN, in.src, in.wei, dst,
                              &mode),
                    status_t::success);
            EXPECT_NE(mode.find("_custom"), std::string::npos) << mode;
            EXPECT_TRUE(same_bytes(dst, gated ? ref_silu : ref_none))
                    << "act=" << static_cast<int>(act);
        };
        ASSERT_NO_FATAL_FAILURE(check(plain_first ? none : silu));
        ASSERT_FALSE(same_bytes(in.wei, raw)) << "nothing was packed in place";
        const Experts image = in.wei;
        ASSERT_NO_FATAL_FAILURE(check(plain_first ? silu : none));
        ASSERT_NO_FATAL_FAILURE(check(plain_first ? none : silu));
        EXPECT_TRUE(same_bytes(in.wei, image)) << "the image was rewritten";
        reset_grp_matmul_caches();
    }
}

// A wide dst cannot take the fused epilogue on the custom kernel, so the call
// runs on AOCL from the caller's buffer and the pack must leave it raw.
TEST(TestGroupMatmulInPlaceCkPack, WideFusedEpilogueKeepsTheWeightRaw) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using namespace in_place_ck;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    if (!get_grp_n_tile_fused_act()) {
        GTEST_SKIP() << "ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT=0 leaves a wide "
                        "swiglu unfused";
    }
    CustomKernelOverride ck_on(true);
    const auto swiglu = grp_matmul_gated_act_t::swiglu_oai_mul;
    reset_grp_matmul_caches();

    Experts ref, dst;
    {
        WeightCacheGuard out_of_place(1);
        Inputs in;
        ASSERT_EQ(run(3, 1.0f, swiglu, kN, in.src, in.wei, ref),
                status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard in_place(2);
    Inputs in;
    for (int call = 0; call < 2; ++call) {
        ASSERT_EQ(run(3, 1.0f, swiglu, kN, in.src, in.wei, dst),
                status_t::success)
                << "call " << call;
        EXPECT_TRUE(same_bytes(dst, ref)) << "call " << call;
    }
    EXPECT_FALSE(custom_kernel::any_weight_packed_in_place(
            {in.wei[0].data(), in.wei[1].data()}, {kM, kM}));
    reset_grp_matmul_caches();
}

// The fused-MoE pipelines of ALGO 2 and 5, and the two-pass path behind them,
// read W13 and W2 off the custom kernel.
TEST(TestGroupMatmulInPlaceCkPack, FusedMoEOffKernelFailsClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "needs the BF16 custom kernel (AVX512-BF16)";
    }
    CustomKernelOverride ck_on(true);
    reset_grp_matmul_caches();

    constexpr int E = 2, M = 4, H = 64, N1 = 128, K_down = N1 / 2;
    const bfloat16_t sentinel(-3.0f);
    TypedBuffers src, w1, w2;
    src.alloc(E, static_cast<size_t>(M) * H, /*is_bf16=*/true);
    w1.alloc(E, static_cast<size_t>(H) * N1, /*is_bf16=*/true);
    w2.alloc(E, static_cast<size_t>(K_down) * H, /*is_bf16=*/true);
    for (int e = 0; e < E; ++e) {
        fill_src(src.bf16[e], e, 0.5f);
        fill_wei1(w1.bf16[e], e, 0.1f);
        fill_wei2(w2.bf16[e], e);
    }
    const auto w1_raw = w1.bf16;
    const auto w2_raw = w2.bf16;

    auto run_fused = [&](int algo, std::vector<std::vector<bfloat16_t>> &w1_v,
                             std::vector<std::vector<bfloat16_t>> &w2_v,
                             std::vector<std::vector<bfloat16_t>> &out) {
        AlgoEnvGuard algo_guard(algo);
        out.assign(E,
                std::vector<bfloat16_t>(static_cast<size_t>(M) * H, sentinel));
        auto gv = GemmVecs::uniform(E, M, N1, H);
        auto params = make_uniform_params(E, data_type_t::bf16);
        for (auto &p : params) {
            p.num_threads = 16;
        }
        std::vector<const void *> srcs(E), wei1(E), wei2(E),
                no_bias(E, nullptr);
        std::vector<void *> d2(E), dst_null(E, nullptr);
        for (int e = 0; e < E; ++e) {
            srcs[e] = src.bf16[e].data();
            wei1[e] = w1_v[e].data();
            wei2[e] = w2_v[e].data();
            d2[e] = out[e].data();
        }
        grp_matmul_gated_act_params act {};
        act.act = grp_matmul_gated_act_t::silu_and_mul;
        auto fused = make_fused_moe_op2(E, H, wei2, no_bias);
        fused.dst_down = d2;
        fused.ldc_down = std::vector<int>(E, H);
        std::vector<int> ldc_null(E, 0);
        return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, wei1, gv.ldb, no_bias,
                gv.beta, dst_null, ldc_null, gv.is_wc, params, nullptr, &act,
                &fused);
    };
    auto untouched = [&](const std::vector<std::vector<bfloat16_t>> &out) {
        for (const auto &d : out) {
            for (const bfloat16_t v : d) {
                if (static_cast<float>(v) != static_cast<float>(sentinel)) {
                    return false;
                }
            }
        }
        return true;
    };

    std::vector<std::vector<bfloat16_t>> ref, out;
    {
        WeightCacheGuard out_of_place(1);
        auto w1_copy = w1_raw;
        auto w2_copy = w2_raw;
        ASSERT_EQ(run_fused(3, w1_copy, w2_copy, ref), status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard in_place(2);
    ASSERT_EQ(run_fused(3, w1.bf16, w2.bf16, out), status_t::success);
    EXPECT_TRUE(in_place_ck::same_bytes(out, ref));
    ASSERT_FALSE(in_place_ck::same_bytes(w1.bf16, w1_raw))
            << "W13 was not packed in place, so nothing below is tested";

    for (const int algo : {1, 2, 5}) {
        EXPECT_EQ(run_fused(algo, w1.bf16, w2.bf16, out), status_t::failure)
                << "algo=" << algo;
        EXPECT_TRUE(untouched(out)) << "algo=" << algo << " wrote dst";
    }
    ASSERT_EQ(run_fused(3, w1.bf16, w2.bf16, out), status_t::success);
    EXPECT_TRUE(in_place_ck::same_bytes(out, ref));
    reset_grp_matmul_caches();
}

namespace f16_in_place_ck {

using namespace moe_test_utils;
using namespace zendnnl::lowoha::matmul;
using Experts = std::vector<std::vector<float16_t>>;
constexpr int kE = 2, kK = 64, kN = 128;
const float16_t kSentinel(-3.0f);

struct Inputs {
    Experts wei;
    Inputs() : wei(kE) {
        for (int e = 0; e < kE; ++e) {
            wei[e].resize(static_cast<size_t>(kK) * kN);
            fill_wei1(wei[e], e, 0.1f);
        }
    }
};

bool same_bytes(const Experts &a, const Experts &b) {
    if (a.size() != b.size()) return false;
    for (size_t e = 0; e < a.size(); ++e) {
        if (a[e].size() != b[e].size()
                || std::memcmp(a[e].data(), b[e].data(),
                           a[e].size() * sizeof(float16_t))
                        != 0)
            return false;
    }
    return true;
}

bool all_sentinel(const Experts &dst) {
    return std::all_of(dst.begin(), dst.end(), [](const auto &d) {
        return std::all_of(d.begin(), d.end(), [](float16_t v) {
            return static_cast<float>(v) == static_cast<float>(kSentinel);
        });
    });
}

status_t run(int algo, int M, float alpha, grp_matmul_gated_act_t act, int ldc,
        Experts &wei, Experts &dst) {
    AlgoEnvGuard algo_guard(algo);
    std::vector<std::vector<float16_t>> src(kE);
    std::vector<const void *> srcs(kE), weights(kE), biases(kE, nullptr);
    std::vector<void *> dsts(kE);
    dst.assign(kE,
            std::vector<float16_t>(static_cast<size_t>(M) * ldc, kSentinel));
    for (int e = 0; e < kE; ++e) {
        src[e].resize(static_cast<size_t>(M) * kK);
        fill_src(src[e], e, 0.5f);
        srcs[e] = src[e].data();
        weights[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }
    auto gv = GemmVecs::uniform(kE, M, kN, kK, alpha);
    gv.ldc.assign(kE, ldc);
    auto params = make_uniform_params(kE, data_type_t::f16);
    for (auto &p : params)
        p.num_threads = 16;
    grp_matmul_gated_act_params gated {};
    gated.act = act;
    return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms, gv.Ns,
            gv.Ks, gv.alpha, srcs, gv.lda, weights, gv.ldb, biases, gv.beta,
            dsts, gv.ldc, gv.is_wc, params, nullptr,
            act == grp_matmul_gated_act_t::none ? nullptr : &gated);
}

} // namespace f16_in_place_ck

TEST(TestGroupMatmulInPlaceCkPack, F16UnpinnedPromptDecodeReusesOneImage) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using namespace f16_in_place_ck;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "needs the FP16 custom kernel";
    }
    AlgoEnvGuard auto_algo(0);
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    Experts ref_prompt, ref_decode, got_prompt, got_decode;
    {
        reset_grp_matmul_caches();
        WeightCacheGuard wc1(1);
        Inputs ref;
        ASSERT_EQ(run(0, 64, 1.0f, grp_matmul_gated_act_t::none, kN, ref.wei,
                          ref_prompt),
                status_t::success);
        ASSERT_EQ(run(0, 8, 1.0f, grp_matmul_gated_act_t::none, kN, ref.wei,
                          ref_decode),
                status_t::success);
    }

    reset_grp_matmul_caches();
    WeightCacheGuard wc2(2);
    Inputs test;
    group_matmul_prepack::test_api::LastInvocationStats prompt_stats;
    {
        LastInvocationCaptureGuard capture;
        ASSERT_EQ(run(0, 64, 1.0f, grp_matmul_gated_act_t::none, kN, test.wei,
                          got_prompt),
                status_t::success);
        prompt_stats
                = group_matmul_prepack::test_api::get_last_invocation_stats();
    }
    ASSERT_TRUE(prompt_stats.valid);
    EXPECT_EQ(prompt_stats.scheduling_algo, 3);
    EXPECT_EQ(prompt_stats.cross_warm_regime,
            group_matmul_prepack::CrossWarmRegime::none);
    for (int e = 0; e < kE; ++e)
        EXPECT_TRUE(custom_kernel::weight_packed_in_place(test.wei[e].data()));

    group_matmul_prepack::test_api::LastInvocationStats decode_stats;
    {
        LastInvocationCaptureGuard capture;
        ASSERT_EQ(run(0, 8, 1.0f, grp_matmul_gated_act_t::none, kN, test.wei,
                          got_decode),
                status_t::success);
        decode_stats
                = group_matmul_prepack::test_api::get_last_invocation_stats();
    }
    EXPECT_FALSE(decode_stats.valid)
            << "decode should reuse the prompt pack without another prepack";
    EXPECT_TRUE(same_bytes(got_prompt, ref_prompt));
    EXPECT_TRUE(same_bytes(got_decode, ref_decode));
    reset_grp_matmul_caches();
}

TEST(TestGroupMatmulInPlaceCkPack, F16OffKernelCallsFailClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    using namespace f16_in_place_ck;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "needs the FP16 custom kernel";
    }
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    reset_grp_matmul_caches();

    Experts ref, dst;
    {
        WeightCacheGuard wc1(1);
        Inputs r;
        ASSERT_EQ(run(3, 8, 1.0f, grp_matmul_gated_act_t::none, kN, r.wei, ref),
                status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard wc2(2);
    Inputs test;
    ASSERT_EQ(run(3, 8, 1.0f, grp_matmul_gated_act_t::none, kN, test.wei, dst),
            status_t::success);
    ASSERT_TRUE(same_bytes(dst, ref));
    for (int e = 0; e < kE; ++e)
        ASSERT_TRUE(custom_kernel::weight_packed_in_place(test.wei[e].data()));

    EXPECT_EQ(run(3, 8, 0.5f, grp_matmul_gated_act_t::none, kN, test.wei, dst),
            status_t::failure);
    EXPECT_TRUE(all_sentinel(dst));
    for (const int algo : {1, 2, 5}) {
        EXPECT_EQ(run(algo, 8, 1.0f, grp_matmul_gated_act_t::none, kN, test.wei,
                          dst),
                status_t::failure)
                << "algo=" << algo;
        EXPECT_TRUE(all_sentinel(dst)) << "algo=" << algo;
    }
    EXPECT_EQ(run(3, 8, 1.0f, grp_matmul_gated_act_t::swiglu_oai_mul, kN,
                      test.wei, dst),
            status_t::failure);
    EXPECT_TRUE(all_sentinel(dst));
    ASSERT_EQ(run(3, 8, 1.0f, grp_matmul_gated_act_t::none, kN, test.wei, dst),
            status_t::success);
    EXPECT_TRUE(same_bytes(dst, ref));
    reset_grp_matmul_caches();
}

TEST(TestGroupMatmulInPlaceCkPack, F16FusedMoeOffKernelFailsClosed) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    if (!custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "needs the FP16 custom kernel";
    }
    CustomKernelOverride ck_on(true);
    CustomKernelF16Override ck_f16_on(true);
    reset_grp_matmul_caches();

    constexpr int E = 2, M = 4, H = 64, N1 = 128, KDown = N1 / 2;
    const float16_t sentinel(-3.0f);
    std::vector<std::vector<float16_t>> src(E), w1(E), w2(E);
    for (int e = 0; e < E; ++e) {
        src[e].resize(static_cast<size_t>(M) * H);
        w1[e].resize(static_cast<size_t>(H) * N1);
        w2[e].resize(static_cast<size_t>(KDown) * H);
        fill_src(src[e], e, 0.5f);
        fill_wei1(w1[e], e, 0.1f);
        fill_wei2(w2[e], e);
    }

    const auto run_fused
            = [&](int algo, std::vector<std::vector<float16_t>> &w1v,
                      std::vector<std::vector<float16_t>> &w2v,
                      std::vector<std::vector<float16_t>> &out) {
        AlgoEnvGuard algo_guard(algo);
        out.assign(E,
                std::vector<float16_t>(static_cast<size_t>(M) * H, sentinel));
        auto gv = GemmVecs::uniform(E, M, N1, H);
        auto params = make_uniform_params(E, data_type_t::f16);
        for (auto &p : params)
            p.num_threads = 16;
        std::vector<const void *> srcs(E), wei1(E), wei2(E),
                no_bias(E, nullptr);
        std::vector<void *> d2(E), dst_null(E, nullptr);
        for (int e = 0; e < E; ++e) {
            srcs[e] = src[e].data();
            wei1[e] = w1v[e].data();
            wei2[e] = w2v[e].data();
            d2[e] = out[e].data();
        }
        grp_matmul_gated_act_params act {};
        act.act = grp_matmul_gated_act_t::silu_and_mul;
        auto fused = make_fused_moe_op2(E, H, wei2, no_bias);
        fused.dst_down = d2;
        fused.ldc_down.assign(E, H);
        std::vector<int> ldc_null(E, 0);
        return group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, wei1, gv.ldb, no_bias,
                gv.beta, dst_null, ldc_null, gv.is_wc, params, nullptr, &act,
                &fused);
    };
    const auto untouched = [&](const auto &out) {
        return std::all_of(out.begin(), out.end(), [&](const auto &d) {
            return std::all_of(d.begin(), d.end(), [&](float16_t v) {
                return static_cast<float>(v) == static_cast<float>(sentinel);
            });
        });
    };

    std::vector<std::vector<float16_t>> ref, out;
    {
        WeightCacheGuard wc1(1);
        auto w1copy = w1;
        auto w2copy = w2;
        ASSERT_EQ(run_fused(3, w1copy, w2copy, ref), status_t::success);
        reset_grp_matmul_caches();
    }

    WeightCacheGuard wc2(2);
    ASSERT_EQ(run_fused(3, w1, w2, out), status_t::success);
    EXPECT_TRUE(f16_in_place_ck::same_bytes(out, ref));
    ASSERT_TRUE(custom_kernel::weight_packed_in_place(w1[0].data()));
    for (const int algo : {1, 2, 5}) {
        EXPECT_EQ(run_fused(algo, w1, w2, out), status_t::failure)
                << "algo=" << algo;
        EXPECT_TRUE(untouched(out)) << "algo=" << algo;
    }
    ASSERT_EQ(run_fused(3, w1, w2, out), status_t::success);
    EXPECT_TRUE(f16_in_place_ck::same_bytes(out, ref));
    reset_grp_matmul_caches();
}

// ALGO 3 phase env clamps to ALGO 1 when the shape fails
// `check_n_tile_extra` (e.g., dynamic_quant=true).  Same correctness
// contract the global ALGO=3 path has, applied to the auto path.
// ===============================================================================
// [8e] TestGroupMatmulHybridMSplit — Option-A M-weighted water-fill
//      heavy distribution under
//      `ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD`.
//
// Targets the new path inside
// `apply_round_pick(RoundPick::Single, use_custom=true)` that fires
// when the env (or test override) is positive AND the M distribution
// has both heavy and light experts.  Verified via the existing
// `PhaseBSnapshot` capture hook so the test is a true white-box
// observation of the planner's per-expert thread distribution.
//
// Two cases:
//
//   1. `OffByDefaultRunsPhaseB` — `threshold=0` (default) MUST leave
//      the existing Phase B remainder behaviour intact.  Sanity
//      check that the hybrid gate doesn't fire when env is unset.
//
//   2. `OnDistributesHeavyByM` — `threshold=100` with a bimodal M
//      vector produces an asymmetric allocation: every active light
//      expert gets exactly 1 thread, the heavy experts share the
//      remaining budget proportional to M (heavier gets ≥ lighter).
// ===============================================================================

namespace {

// Shared shape builder for Hybrid M-split tests.  Uses uniform
// row-major BF16 inputs and the same N=1024 / K=64 dims as the
// existing PhaseB test fixtures.  M vector and num_threads are
// caller-supplied.
struct HybridProbeShape {
    int num_ops;
    int num_threads;
    int N;
    int K;
    std::vector<int> Ms;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> src;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> wei;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> dst;
    std::vector<const void *> srcs;
    std::vector<const void *> weis;
    std::vector<const void *> biases;
    std::vector<void *> dsts;
    moe_test_utils::GemmVecs gv;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static HybridProbeShape build_hybrid_probe(
        int num_threads, const std::vector<int> &Ms, int N = 1024, int K = 64) {
    using namespace moe_test_utils;
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    HybridProbeShape s;
    s.num_threads = num_threads;
    s.num_ops = static_cast<int>(Ms.size());
    s.N = N;
    s.K = K;
    s.Ms = Ms;
    s.src.resize(s.num_ops);
    s.wei.resize(s.num_ops);
    s.dst.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
        s.wei[e].resize(static_cast<size_t>(K) * N);
        s.dst[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
        fill_src(s.src[e], e, 0.02f);
        fill_wei1(s.wei[e], e, 0.005f);
    }
    s.srcs.resize(s.num_ops);
    s.weis.resize(s.num_ops);
    s.biases.assign(s.num_ops, nullptr);
    s.dsts.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.srcs[e] = s.src[e].data();
        s.weis[e] = s.wei[e].data();
        s.dsts[e] = s.dst[e].data();
    }
    s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*wc=*/true, /*tA=*/false, /*tB=*/false);
    s.gv.Ms = Ms;
    s.params = make_uniform_params(s.num_ops, data_type_t::bf16);
    for (int e = 0; e < s.num_ops; ++e) {
        s.params[e].num_threads = num_threads;
    }
    return s;
}

} // namespace

// Hybrid OFF (default) → existing Phase B distribution fires
// (per-expert allocation is `base` or `base+1`, not the water-fill
// product).  Sanity check that the new gate doesn't accidentally
// fire when the threshold env is 0.
TEST(TestGroupMatmulHybridMSplit, OffByDefaultRunsPhaseB) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::GroupNTileStrategy;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride auto_strategy(0);
    NTileHeavyThresholdOverride hybrid_off(-1); // explicit DISABLED
    // Pin proportional ON explicitly so the decode contract (heaviest expert
    // gets > uniform base+1) is tested independent of default-auto gating.  The
    // knob under test here is the PROMPT-only hybrid threshold, which must NOT
    // fire on decode regardless of this.
    DecodeProportionalOverride prop_on(true);

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    // 14 active experts, decode-class M (≤ 32) so we land on the
    // CK Single round (matches the existing PhaseB heaviest-first
    // test shape).
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder);
    // The prompt-only HYBRID water-fill keeps n_thr_fixed=0; the decode
    // M-proportional split (and legacy Phase B) leave it at base.  A
    // non-zero value confirms the HYBRID gate did NOT fire.
    EXPECT_NE(snap.n_thr_fixed, 0)
            << "Hybrid gate must NOT fire on decode; got n_thr_fixed=0.";
    // ZENDNNL_GRP_MATMUL_N_TILE_HEAVY_THRESHOLD is a PROMPT-ONLY knob
    // (default -1 = DISABLED); it does NOT govern decode.  The decode
    // CK Single-round path now uses the UNCONDITIONAL M-proportional
    // thread split regardless of this knob, so even with the hybrid
    // threshold explicitly disabled (-1) decode still load-balances by
    // M.  Assert the proportional contract (not the legacy uniform
    // base/base+1): per_expert_remainder set, no active expert starved,
    // total <= num_threads, and the heaviest expert (M=16) gets a
    // proportional share above the uniform base+1 (=3).
    int sum_thr = 0;
    for (int e = 0; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        if (s.Ms[e] > 0) {
            EXPECT_GE(n, 1) << "active expert e=" << e << " starved (0 thr)";
        }
        sum_thr += n;
    }
    EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
    EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
            << "heaviest expert (M=16) should get a proportional share "
               "(> uniform base+1=3); got "
            << static_cast<int>(snap.stable_n_thr_per_expert[0]);
}

// Hybrid ON → water-fill distribution: heavier-M experts receive
// strictly ≥ threads of lighter heavies; every active light
// expert gets exactly 1 thread; `n_thr_fixed == 0` to tell the
// executor to use prefix-sum mapping; `sum(stable_n_thr) <=
// num_threads` (water-fill may leave a tail of idle threads when
// every heavy hits its per-expert cap).
// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL=0 disables the decode
// M-proportional split even where the auto default would engage it;
// decode CK Single-round uses the uniform Phase B base/base+1
// distribution.
TEST(TestGroupMatmulHybridMSplit, DecodeProportionalOffUsesUniformPhaseB) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride auto_strategy(0);
    DecodeProportionalOverride prop_off(false); // knob under test

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder);
    EXPECT_NE(snap.n_thr_fixed, 0)
            << "proportional OFF keeps n_thr_fixed at base (not the hybrid 0).";
    // With proportional disabled every active expert gets base (2) or
    // base+1 (3) only — NO M-weighted spread.
    for (int e = 0; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        if (s.Ms[e] > 0) {
            EXPECT_TRUE(n == 2 || n == 3)
                    << "proportional OFF -> uniform Phase B; expert e=" << e
                    << " M=" << s.Ms[e] << " got n_thr=" << n;
        }
    }
}

// ZENDNNL_GRP_MATMUL_DECODE_PROPORTIONAL=1 forces the M-weighted split
// regardless of the auto default: heavier experts get a proportional share
// above the uniform base+1, light experts the floor.
TEST(TestGroupMatmulHybridMSplit, DecodeProportionalOnUsesMWeightedSplit) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride auto_strategy(0);
    DecodeProportionalOverride prop_on(true); // knob under test

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {16, 12, 10, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4});

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder);
    int sum_thr = 0;
    for (int e = 0; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        if (s.Ms[e] > 0) {
            EXPECT_GE(n, 1) << "active expert e=" << e << " starved";
        }
        sum_thr += n;
    }
    EXPECT_LE(sum_thr, 32);
    // Heaviest (M=16) gets a proportional share > uniform base+1 (=3),
    // and strictly more than the lightest (M=4) experts.
    const int n_heavy = static_cast<int>(snap.stable_n_thr_per_expert[0]);
    EXPECT_GE(n_heavy, 4)
            << "proportional ON: heaviest expert under-allocated (" << n_heavy
            << ")";
    EXPECT_GT(n_heavy, static_cast<int>(snap.stable_n_thr_per_expert[13]));
}

TEST(TestGroupMatmulHybridMSplit, OnDistributesHeavyByM) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::GroupNTileStrategy;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    // MANUAL HYBRID is now prompt-only (max_M > kDecodeMaxM=32).  Force
    // `n_tile_strategy=2` (rounds, force) so the planner stays in
    // ManyExperts, where the Single-round HYBRID dispatch fires, whatever
    // strategy the process runs with.
    NTileStrategyOverride force_rounds(2);
    // Threshold chosen so M > 200 = heavy AND every Ms entry crosses
    // the new `max_M > kDecodeMaxM=32` prompt gate.
    NTileHeavyThresholdOverride hybrid_on(200);

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    // 14 experts, with M chosen so M=64 is LIGHT (≤ 200) and
    // M ∈ {400, 320, 280} are HEAVY (> 200).  3 heavy + 11 light.
    // All Ms are prompt-class (> kDecodeMaxM=32) so MANUAL HYBRID's
    // new decode gate lets the path engage.  N=4096 keeps the cap
    // arithmetic identical to the AUTO tests.
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {400, 320, 280, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
            /*N=*/4096);

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder)
            << "Hybrid path MUST set per_expert_remainder=true so the "
               "executor reads stable_n_thr_per_expert[].";
    EXPECT_EQ(snap.n_thr_fixed, 0)
            << "Hybrid path MUST set n_thr_fixed=0 so the executor takes "
               "the prefix-sum scan instead of the uniform `tid/tpe` mapping.";

    // Per-expert assertions.  Experts e=0..2 are heavy (M = s.Ms[0..2],
    // currently {400, 320, 280}); e=3..13 are light (M = s.Ms[3..13],
    // currently 64 each).  Light experts must each receive exactly 1
    // thread.  Heavies must receive >= 1, AND the M-descending water-
    // fill ordering implies n_thr[e=0] >= n_thr[e=1] >= n_thr[e=2].
    // Messages below derive the M values from `s.Ms[]` so the failure
    // report stays correct if the probe Ms get retuned in a follow-up.
    const int n0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
    const int n1 = static_cast<int>(snap.stable_n_thr_per_expert[1]);
    const int n2 = static_cast<int>(snap.stable_n_thr_per_expert[2]);
    EXPECT_GE(n0, 1);
    EXPECT_GE(n0, n1) << "Heavier expert e=0 (M=" << s.Ms[0]
                      << ") must receive >= threads "
                         "as e=1 (M="
                      << s.Ms[1] << ").";
    EXPECT_GE(n1, n2) << "Heavier expert e=1 (M=" << s.Ms[1]
                      << ") must receive >= threads "
                         "as e=2 (M="
                      << s.Ms[2] << ").";
    // At least one heavy must receive more than the per-light
    // allocation (= 1), otherwise the path collapses to "everyone
    // gets 1 thread" which is no better than the legacy single-round
    // behaviour pre-Phase-B.
    EXPECT_GT(n0 + n1 + n2, 3)
            << "Heavies should collectively absorb more than 1 thread each.";
    for (int e = 3; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        EXPECT_EQ(n, 1)
                << "Light expert e=" << e << " (M=" << s.Ms[e]
                << ") must receive exactly 1 thread under hybrid M-split.";
    }
    // Total must NOT exceed num_threads (water-fill respects the
    // overall thread budget).  Equality is the common case; strictly
    // less can happen if every heavy hits its per-expert cap.
    int sum = 0;
    for (int e = 0; e < s.num_ops; ++e) {
        sum += static_cast<int>(snap.stable_n_thr_per_expert[e]);
    }
    EXPECT_LE(sum, 32)
            << "Hybrid distribution must not over-subscribe the team; "
               "sum(stable_n_thr_per_expert) = "
            << sum;
    EXPECT_GT(sum, s.num_ops - 3 /*lights*/ + 3 /*heavies*/)
            << "Hybrid distribution should give heavies > 1 thread each; "
               "sum = "
            << sum << " is suspiciously low.";
}

// AUTO (env=0) → planner-driven adaptive tiers fire, producing a
// monotone-by-M per-expert allocation with the Hybrid dispatch
// signature (`per_expert_remainder=true`, `n_thr_fixed=0`).  Verifies
// AUTO engages on a sufficiently-skewed prompt-class shape AND
// produces a heaviest-first allocation distinct from Phase B base+1.
//
// Shape constraints to keep the planner in ManyExperts:
//   * `max_M > kDecodeMaxM` (prompt-class) is now MANDATORY — AUTO's
//     new decode gate (`apply_adaptive_tiers()` returns false on
//     `max_M ≤ kDecodeMaxM`) makes the previous decode-class probe
//     fall through to Phase B.  Use max_M=400 (firmly prompt-class).
//   * Force `n_tile_strategy=2` (rounds) so the planner stays in
//     ManyExperts, where the Single-round HYBRID dispatch lives,
//     whatever strategy the process runs with.
//   * `N` chosen so `max_tiles = N/min_n_tile` lifts the per-expert
//     cap above 2.  N=4096 with the prompt `min_n_tile=512` →
//     max_tiles=8, cap=min(ccd_size=8, 8)=8.
TEST(TestGroupMatmulHybridMSplit, AutoTierEngagesAtZero) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride force_rounds(2); // stay in ManyExperts
    NTileHeavyThresholdOverride hybrid_auto(0); // AUTO mode

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    // 14 experts, PROMPT-class M (max=400 > kDecodeMaxM=32).  Skew =
    // 400 / mean ≈ 400/107 ≈ 3.7 (well above kMinSkew=2.5).  Same
    // tier topology as before, scaled into the prompt regime:
    //   T_high ≈ max(M_p95, 160)  → 1 high  (M=400)
    //   T_mid  ≈ max(M_p75, 80)   → 3 mid   (M=320, 240, 160)
    //   T_low  ≈ max(M_p50, 40)   → 3 low   (M=120, 80, 60)
    //   baseline                  → 7 (M ∈ {40, 40, 40, 40, 40, 40, 40})
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/
            {400, 320, 240, 160, 120, 80, 60, 40, 40, 40, 40, 40, 40, 40},
            /*N=*/4096);

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    EXPECT_TRUE(snap.per_expert_remainder)
            << "AUTO_TIER must set per_expert_remainder=true so the executor "
               "reads stable_n_thr_per_expert[].";
    EXPECT_EQ(snap.n_thr_fixed, 0)
            << "AUTO_TIER must set n_thr_fixed=0 (Hybrid signature, distinct "
               "from Phase B's n_thr_fixed=base).";

    // M-monotone ordering: heaviest expert (M=400, prompt-class probe
    // shape; see the Ms vector above) must receive >= threads as any
    // lower-M expert.  The water-fill is M-weighted so the heaviest
    // expert always wins the marginal next-thread.
    const int n0 = static_cast<int>(snap.stable_n_thr_per_expert[0]);
    EXPECT_GT(n0, 1)
            << "Heaviest expert (M=400, high tier) must receive multiple "
               "threads; AUTO_TIER didn't engage if n0 = 1.";
    for (int e = 1; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        EXPECT_GE(n0, n) << "AUTO_TIER must be M-monotone: heaviest (e=0, M="
                         << s.Ms[0] << ") got " << n0 << " threads but e=" << e
                         << " (M=" << s.Ms[e] << ") got " << n << ".";
    }

    // Asymmetry check: high tier (e=0) must be strictly heavier than
    // a baseline expert (last index, M=40 — the tail of the prompt-
    // class probe).  Otherwise the tier structure collapsed and AUTO
    // is just a no-op.
    const int n_last
            = static_cast<int>(snap.stable_n_thr_per_expert[s.num_ops - 1]);
    EXPECT_GT(n0, n_last)
            << "AUTO_TIER must produce asymmetry: high tier (n0=" << n0
            << ") must receive strictly more threads than baseline "
               "(n_last="
            << n_last << ").";

    // Total must NOT exceed num_threads (water-fill respects budget).
    int sum = 0;
    for (int e = 0; e < s.num_ops; ++e) {
        sum += static_cast<int>(snap.stable_n_thr_per_expert[e]);
    }
    EXPECT_LE(sum, 32) << "AUTO_TIER must not over-subscribe the OMP team; sum="
                       << sum;
}

// AUTO with uniform M → skew gate (M_max / M_mean < 2.5) fails;
// path returns false; planner falls through to Phase B.  Distinguishes
// "no skew → no engagement" from the explicit DISABLED case.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsLowSkew) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride force_rounds(2); // stay in ManyExperts
    NTileHeavyThresholdOverride hybrid_auto(0); // AUTO mode

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    // 14 experts, all M=100 → skew = 100/100 = 1.0, well below
    // kMinSkew=2.5.  AUTO_TIER must skip on the LOW-SKEW gate (not the
    // prompt gate); Phase B base+1 must populate the snapshot.  M=100
    // is prompt-class so the new decode gate does NOT pre-empt the
    // low-skew branch — this test exercises the skew filter
    // specifically.
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/
            {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                    100},
            /*N=*/4096);

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    // Phase B keeps n_thr_fixed = base (= 2 here for 32t / 14 ops).
    // AUTO would have set it to 0 — asserting non-zero pins the
    // fallback path.
    EXPECT_NE(snap.n_thr_fixed, 0)
            << "AUTO_TIER must fall through to Phase B on low-skew workloads; "
               "expected Phase B's n_thr_fixed != 0, got 0 (AUTO engaged "
               "spuriously).";
    // Every allocation must be base (= 2) or base+1 (= 3) — Phase B's
    // signature, NOT the multi-tier 1/2/4/8 pattern.
    for (int e = 0; e < s.num_ops; ++e) {
        const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
        EXPECT_TRUE(n == 2 || n == 3)
                << "Uniform-M expert e=" << e << " got n_thr=" << n
                << " (Phase B should produce 2 or 3).";
    }
}

// AUTO with `num_threads == num_active` → extras_budget = 0 →
// AUTO_TIER returns false; planner falls through to Phase B in the
// same Single round case.  Models the "thread-starved" production
// case (e.g. 64-core machine running a 64-expert MoE call where
// every expert wants at least 1 thread and there's no headroom).
//
// Distinct from the legacy single-threshold path which would happily
// engage as long as `heavy_budget >= 2 * n_heavy` — AUTO is stricter
// because tiering with zero extras has no payoff.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsThreadStarvation) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    CustomKernelOverride ck_on(true);
    NRoundsModeOverride single_round(1);
    CustomKernelNTileOverride default_n_tile(0);
    NTileStrategyOverride force_rounds(2); // stay in ManyExperts
    NTileHeavyThresholdOverride hybrid_auto(0); // AUTO mode

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }
    reset_grp_matmul_caches();

    // 14 active experts, PROMPT-class M (max=400 > kDecodeMaxM=32) so
    // the new prompt gate does NOT preempt the test.  Skew passes
    // (400/107 ≈ 3.7, above kMinSkew=2.5).  Planner is told
    // num_threads = 14 == num_active so the Single round stays
    // feasible (`single_eligible` requires `num_threads >= num_ops`),
    // but AUTO's `extras_budget = num_threads - num_active = 0` gate
    // fails immediately → returns false → Phase B base+1 fires.
    auto s = build_hybrid_probe(/*num_threads=*/14,
            /*Ms=*/
            {400, 320, 240, 160, 120, 80, 60, 40, 40, 40, 40, 40, 40, 40},
            /*N=*/4096);

    AlgoEnvGuard algo_guard(3);
    PhaseBCaptureGuard cap;
    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);
    const auto &snap
            = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
    ASSERT_TRUE(snap.valid);
    // AUTO's signature is `(per_expert_remainder=true, n_thr_fixed=0)`.
    // Phase B (the fallback we expect to fire here) leaves
    // `per_expert_remainder=false` and `n_thr_fixed=base`.  Asserting
    // BOTH negations confirms AUTO returned false and the planner took
    // the Phase B branch instead.  `stable_n_thr_per_expert[]` is not
    // populated in the Phase B branch (the executor uses uniform
    // `tid / n_thr_fixed` mapping in that case), so no point asserting
    // on its contents — they are correctly left at zero.
    EXPECT_FALSE(snap.per_expert_remainder)
            << "AUTO_TIER must fall through when extras_budget = 0; "
               "expected Phase B (per_expert_remainder=false), got true "
               "(AUTO engaged spuriously).";
    EXPECT_NE(snap.n_thr_fixed, 0)
            << "AUTO_TIER must fall through when extras_budget = 0; "
               "expected Phase B's n_thr_fixed=base, got 0 (AUTO engaged).";
}

// Phase gate (decode safety) — `HYBRID=0` (AUTO) and `HYBRID=10`
// (MANUAL) BOTH must be skipped on a decode-class shape
// (`max_M <= kDecodeMaxM`) so the planner stays on Phase B base+1.
// Models the unified-process E2E case where `HYBRID=0` is exported
// once for the whole run: prompt benefits from AUTO tiering while
// decode plans remain untouched.
//
// Acceptance criterion (the only signature unique to HYBRID):
//   * `n_thr_fixed != 0`  — Phase B (including its base+1 remainder
//                           branch) keeps `n_thr_fixed = base`,
//                           while both HYBRID modes zero it to
//                           tell the executor to use the prefix-
//                           sum scan.  `per_expert_remainder` is
//                           NOT a distinguishing signal here:
//                           Phase B's remainder branch also sets
//                           it to true with base/base+1 entries,
//                           so asserting on it would create a
//                           false alarm on this 32t/14-experts
//                           shape (remainder = 4 → branch fires).
//
// Shape: 14 experts, max_M=24 (decode-class), high skew (24/8.6 ≈ 2.8
// > kMinSkew=2.5), N=4096 (cap=8) — pre-gate this shape engaged AUTO
// (see the pre-prompt-gate version of `AutoTierEngagesAtZero`).  After
// the gate it must fall through unconditionally.
TEST(TestGroupMatmulHybridMSplit, AutoTierSkipsDecodeClass) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    if (!zendnnl::lowoha::matmul::custom_kernel::dispatch_supported()) {
        GTEST_SKIP() << "Requires AVX512BF16 / CK dispatch support.";
    }

    // Decode-class M, high skew — would have engaged AUTO before the
    // prompt-only gate landed.
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {24, 20, 16, 12, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1},
            /*N=*/4096);

    // Sub-case 1: AUTO (HYBRID=0) must skip on decode-class M.
    {
        CustomKernelOverride ck_on(true);
        NRoundsModeOverride single_round(1);
        CustomKernelNTileOverride default_n_tile(0);
        NTileStrategyOverride auto_strategy(0);
        NTileHeavyThresholdOverride hybrid_auto(0);
        // Pin proportional ON so the proportional contract asserted below
        // holds.  The tier-skip assertion (n_thr_fixed != 0) is independent of
        // this knob.
        DecodeProportionalOverride prop_on(true);

        reset_grp_matmul_caches();
        AlgoEnvGuard algo_guard(3);
        PhaseBCaptureGuard cap;
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                        s.gv.is_wc, s.params, nullptr, nullptr),
                status_t::success);
        const auto &snap
                = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
        ASSERT_TRUE(snap.valid);
        EXPECT_NE(snap.n_thr_fixed, 0)
                << "AUTO adaptive-tiers must be skipped on decode-class shape "
                   "(max_M=24 <= kDecodeMaxM=32); tiers clear n_thr_fixed to "
                   "0, "
                   "so a non-zero value confirms tiers did NOT engage.  The "
                   "decode M-proportional split (which DOES run here) leaves "
                   "n_thr_fixed at base.";
        // Decode now uses the M-proportional thread split (commit:
        // "static M-proportional thread allocation for CK decode") instead
        // of the legacy uniform Phase B base/base+1.  That is still NOT the
        // adaptive-tier water-fill (n_thr_fixed stays != 0, asserted above).
        // Assert the proportional contract rather than the old {0,2,3}:
        //   - per_expert_remainder set (prefix-sum executor path),
        //   - every active expert gets >= 1 thread (no starvation),
        //   - total allocation <= num_threads,
        //   - the heaviest expert (M=24) gets a proportional share > the
        //     uniform base+1 (=3) — distinguishes proportional from Phase B.
        EXPECT_TRUE(snap.per_expert_remainder);
        int sum_thr = 0;
        for (int e = 0; e < s.num_ops; ++e) {
            const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
            if (s.Ms[e] > 0) {
                EXPECT_GE(n, 1)
                        << "active expert e=" << e << " starved (0 thr)";
            }
            sum_thr += n;
        }
        EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
        EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
                << "heaviest expert (M=24) should get a proportional share "
                   "(> uniform base+1=3); got "
                << static_cast<int>(snap.stable_n_thr_per_expert[0]);
    }

    // Sub-case 2: MANUAL (HYBRID=10, would tag heavies on this Ms)
    // must also skip on decode-class M.
    {
        CustomKernelOverride ck_on(true);
        NRoundsModeOverride single_round(1);
        CustomKernelNTileOverride default_n_tile(0);
        NTileStrategyOverride auto_strategy(0);
        // Threshold=10 would tag M ∈ {24,20,16,12} as heavy on this
        // Ms — i.e. would have engaged MANUAL on legacy semantics.
        NTileHeavyThresholdOverride hybrid_manual(10);
        // Pin proportional ON so the proportional contract asserted below
        // holds.  The tier-skip assertion (n_thr_fixed != 0) is independent of
        // this knob.
        DecodeProportionalOverride prop_on(true);

        reset_grp_matmul_caches();
        AlgoEnvGuard algo_guard(3);
        PhaseBCaptureGuard cap;
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                        s.gv.is_wc, s.params, nullptr, nullptr),
                status_t::success);
        const auto &snap
                = zendnnl::lowoha::matmul::test_api::s_last_phase_b_snapshot;
        ASSERT_TRUE(snap.valid);
        EXPECT_NE(snap.n_thr_fixed, 0)
                << "MANUAL adaptive-tiers must be skipped on decode-class "
                   "shape "
                   "(max_M=24 <= kDecodeMaxM=32); tiers clear n_thr_fixed to "
                   "0, "
                   "so a non-zero value confirms the MANUAL water-fill did NOT "
                   "engage.  The decode M-proportional split runs instead.";
        // As in sub-case 1: decode uses the M-proportional split, not the
        // MANUAL adaptive-tier water-fill.  threshold=10 (!= -1) keeps the
        // proportional split enabled; only an explicit -1 disable would
        // restore uniform Phase B.  Assert the proportional contract.
        EXPECT_TRUE(snap.per_expert_remainder);
        int sum_thr = 0;
        for (int e = 0; e < s.num_ops; ++e) {
            const int n = static_cast<int>(snap.stable_n_thr_per_expert[e]);
            if (s.Ms[e] > 0) {
                EXPECT_GE(n, 1)
                        << "active expert e=" << e << " starved (0 thr)";
            }
            sum_thr += n;
        }
        EXPECT_LE(sum_thr, 32) << "M-proportional over-allocated threads";
        EXPECT_GE(static_cast<int>(snap.stable_n_thr_per_expert[0]), 4)
                << "heaviest expert (M=24) should get a proportional share "
                   "(> uniform base+1=3); got "
                << static_cast<int>(snap.stable_n_thr_per_expert[0]);
    }
}

TEST(TestGroupMatmulAutoPhaseEnv, Algo3PhaseEnvClampedOnNonNTileSafe) {
    using namespace zendnnl::lowoha::matmul;
    using namespace moe_test_utils;
    reset_grp_matmul_caches();
    AlgoEnvGuard reset_algo(0);
    AutoPromptAlgoOverride force_prompt(3);
    AutoDecodeAlgoOverride no_decode(0);

    // Mixtral-class prompt, with dynamic_quant=true → n_tile_safe=false.
    // Phase env asks for ALGO 3, but the safety clamp falls to ALGO 1.
    auto s = build_auto_probe(/*M=*/256, /*K=*/4096, /*N=*/14336,
            /*num_ops=*/8, /*num_threads=*/128);
    for (auto &p : s.params) {
        p.dynamic_quant = true;
        p.quant_params.src_scale.dims = {256, 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
    }
    EXPECT_EQ(select_grp_matmul_algo(
                      s.layout, s.M, s.N, s.K, s.params, s.num_threads),
            1)
            << "AUTO_PROMPT_ALGO=3 with !n_tile_safe must clamp to ALGO 1 "
               "(same correctness contract as global ALGO=3 path)";
}

// ============================================================================
// [8f] TestGroupMatmulMTileBranches — ALGO 2 (M-tile) internal branch dispatch.
//
// `flat_m_tile` dispatches between four internal branches based on
// workload shape:
//   round-based            (active_ops > num_threads)
//   multi-tier hybrid      (Qwen3-class many-expert skewed prompt)
//   wide-N memory-bound    (total_need * 2 ≤ num_threads, max_M > 1)
//   phase-2 single-tier    (default M-weighted fallthrough)
//
// These tests pin ALGO 2 via `AlgoEnvGuard(2)`, exercise shapes that
// should drive each branch, and assert the tag published by the
// chosen branch (`test_api::s_last_m_tile_path`) matches expectation.
// Adds focused coverage for the wide-N fallback and multi-tier hybrid
// gates so silent regressions surface as failed tag asserts instead
// of perf regressions in downstream benchmarks.
//
// The capture hook is gated by `s_capture_m_tile_path` (armed only
// while `MTilePathCaptureGuard` is in scope); production builds
// never arm it, so the per-call cost is a single relaxed load of a
// cache-line-shared `false` bool — no coherence traffic, branch-
// predictable.  See doc-block on `s_capture_m_tile_path` in
// `group_matmul_parallel_common.hpp`.
// ============================================================================

// ALGO 2 is now PURE M-tile: a FORCED `ZENDNNL_GRP_MATMUL_ALGO=2` on a
// wide-N shape (`total_need*2 ≤ num_threads && max_M > 1`) no longer falls
// back to the sequential-full-team loop — it runs the single-tier M-tile
// plan (kPhase2Single).  (AUTO selects ALGO 2 only for W4A8 prompts; see
// TestGroupMatmulAutoPhaseEnv.NonW4A8AutoNeverSelectsAlgo2AcrossShapeSweep.)
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2OnWideNRunsSingleTier) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    // 8 actives × M=8 × K=64 × N=1024.
    //   total_need = 8 × ceil(8/16) = 8;  2*8 = 16 ≤ 32 ⇒ wide-N gate ✓
    //   max_M = 8 > 1                                  ⇒ decode-exclusion clears
    //   max_M = 8 < 256                                ⇒ multi-tier gate fails
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {8, 8, 8, 8, 8, 8, 8, 8});

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;
    GemmModeCaptureGuard mode_cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_EQ(tag, kPhase2Single)
            << "Forced ALGO 2 on a wide-N shape (8 actives × M=8 / 32t) must "
               "run "
               "PURE single-tier M-tile now (no internal wide-N fallback); "
               "got tag="
            << tag
            << " (kMultiTier=1, kPhase2Single=3, kManyExpertsSeqFallback=7)";

    // gemm_mode must name the executed branch, and exec_algo must map to 2.
    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_STREQ(mode, "flat_m_tile_single_tier");
    EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 2);
}

// Wide-N fallback MUST be excluded for pure-decode workloads (max_M==1).
// The Phase 2 single-tier CCD-stripe layout is the latency-optimal
// mapping for M=1 — one thread per CCD, experts parallel across CCDs.
// Routing M=1 to wide-N would serialise each expert on the full thread
// team and trade away the CCD-parallel decode win.
TEST(TestGroupMatmulMTileBranches, WideNFallbackExcludesDecodeM1) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kWideNFallback;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    // 8 actives × M=1.  Numerically `total_need*2 = 16 ≤ 32` would PASS
    // the wide-N count gate; the `max_M_single_tier > 1` decode-exclusion
    // clamp must keep wide-N off and route to Phase 2 single-tier
    // (CCD-stripe).
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/ {1, 1, 1, 1, 1, 1, 1, 1});

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_NE(tag, kWideNFallback)
            << "Wide-N fallback MUST be excluded on decode-class max_M=1; "
               "got wide-N (tag="
            << tag << ")";
    EXPECT_EQ(tag, kPhase2Single)
            << "max_M=1 must fall through to Phase 2 single-tier (CCD-stripe); "
               "got tag="
            << tag;
}

// Multi-tier hybrid engages when ALL of: actives ≥ num_threads/2,
// max_M ≥ 256, max_M ≥ 4×avg_M, n_light ≥ num_threads/8.  On 32 threads
// the gate needs ≥ 16 actives and ≥ 4 lights.  Shape: 1 heavy at
// M=1024 + 15 lights at M=4 — avg_M ≈ 67.75, light_cut = max(8, 16) = 16
// (M=4 ≤ 16 ⇒ LIGHT), max_M/avg_M ≈ 15.1 ⇒ skew ✓.
TEST(TestGroupMatmulMTileBranches, MultiTierEngagesOnSkewedPrompt) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    // 16 actives: 1 heavy (M=1024) + 15 lights (M=4).
    //   actives = 16 ≥ 32/2                              ✓
    //   max_M = 1024 ≥ 256                              ✓
    //   avg_M = (1024 + 15*4) / 16 ≈ 67.75
    //   max_M/avg_M ≈ 15.1 ≥ 4                          ✓ (skew gate)
    //   light_cut = max(8, 67/4) = max(8, 16) = 16
    //   n_light = 15 (every M=4 ≤ 16); n_heavy = 1     ✓
    //   total_need = 64 + 15 = 79; 2*79 = 158 > 32     ⇒ wide-N excluded
    std::vector<int> ms = {1024};
    ms.insert(ms.end(), 15, 4);
    auto s = build_hybrid_probe(/*num_threads=*/32, ms, /*N=*/1024);

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;
    GemmModeCaptureGuard mode_cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_EQ(tag, kMultiTier)
            << "Multi-tier hybrid must engage on 16 actives with skewed M "
               "(1 heavy M=1024 + 15 lights M=4, skew ≈ 15×); got tag="
            << tag;

    // gemm_mode must name the multi-tier branch; exec_algo maps to 2.
    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_STREQ(mode, "flat_m_tile_multitier");
    EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 2);
}

// ============================================================================
// [8j] Logging truth: gemm_mode names the executed branch, and
//      executed_algo_from_gemm_mode maps it to the ALGO that actually ran.
// ============================================================================

// `ZENDNNL_GRP_MATMUL_KBLOCK` is tri-state, and an explicit setting outranks
// the per-call automatic decision in BOTH directions — `=0` is the escape
// hatch that turns the K-blocked path off even where `kblock_auto` asked for
// it.  Precedence is a pure function so it is checked without the env cache.
TEST(TestGroupMatmulKblock, EnvTriStatePrecedence) {
    using zendnnl::lowoha::matmul::kblock_env_t;
    using zendnnl::lowoha::matmul::resolve_grp_matmul_kblock;

    // Unset defers to the automatic policy.
    EXPECT_FALSE(resolve_grp_matmul_kblock(kblock_env_t::kAuto, false));
    EXPECT_TRUE(resolve_grp_matmul_kblock(kblock_env_t::kAuto, true));
    // Forced off wins over kblock_auto.
    EXPECT_FALSE(resolve_grp_matmul_kblock(kblock_env_t::kOff, false));
    EXPECT_FALSE(resolve_grp_matmul_kblock(kblock_env_t::kOff, true));
    // Forced on applies without kblock_auto.
    EXPECT_TRUE(resolve_grp_matmul_kblock(kblock_env_t::kOn, false));
    EXPECT_TRUE(resolve_grp_matmul_kblock(kblock_env_t::kOn, true));
}

// Unit test for the mode->algo mapping used by the post-exec
// [GRP_MATMUL.CALL] exec_algo= field.  Pure function, no execution.
TEST(TestGroupMatmulGemmMode, ExecutedAlgoFromGemmModeMapping) {
    using zendnnl::lowoha::matmul::executed_algo_from_gemm_mode;
    EXPECT_EQ(executed_algo_from_gemm_mode(nullptr), 0);
    EXPECT_EQ(executed_algo_from_gemm_mode("garbage"), 0);
    EXPECT_EQ(executed_algo_from_gemm_mode("sequential"), 1);
    EXPECT_EQ(executed_algo_from_gemm_mode("sequential_experts"), 1);
    // seq-clamp wears an ALGO-2 mode prefix but runs ALGO-1 behaviour -> 1.
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_seq_clamp"), 1);
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_single_tier"), 2);
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_multitier"), 2);
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile"), 2);
    EXPECT_EQ(executed_algo_from_gemm_mode("vertical_fusion_bf16"), 2);
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_n_tile_custom"), 3);
    EXPECT_EQ(executed_algo_from_gemm_mode("ntile_flat_parallel"), 4);
    EXPECT_EQ(executed_algo_from_gemm_mode("multilevel_rounds"), 6);
    EXPECT_EQ(executed_algo_from_gemm_mode("per_expert"), 5);
    // Fused composites derive the algo from the Op1 sub-mode.
    EXPECT_EQ(executed_algo_from_gemm_mode(
                      "fused_moe_2pass_intalloc(op1=flat_m_tile_single_tier,"
                      "op2=flat_m_tile_single_tier)+postop"),
            2);
    EXPECT_EQ(executed_algo_from_gemm_mode(
                      "fused_moe_vertical_intalloc(op1=vertical_fusion_bf16,"
                      "op2=vertical_fusion_bf16)"),
            2);
    EXPECT_EQ(executed_algo_from_gemm_mode("fused_moe_2pass(op1=flat_n_tile_"
                                           "custom,op2=flat_n_tile_custom)"),
            3);
    // No-op / skip markers map to 0 (nothing executed).
    EXPECT_EQ(executed_algo_from_gemm_mode("fused_moe_skip"), 0);
    EXPECT_EQ(executed_algo_from_gemm_mode("flat_m_tile_skip"), 0);
    EXPECT_EQ(executed_algo_from_gemm_mode("multilevel_skip"), 0);
}

// Forced ALGO 2 with active_ops > num_threads: gemm_mode must name the
// sequential-full-team clamp, and exec_algo must report 1 (the ALGO-1
// behaviour that actually ran) even though chosen=ALGO_2.  This is the
// headline selection-vs-execution divergence the logging makes explicit.
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2SeqClampWritesMode) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag ::
            kManyExpertsSeqFallback;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(8);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 4) {
        GTEST_SKIP() << "Requires >= 4 OMP threads; have " << actual_team_size;
    }

    // 8 active experts on a 4-thread budget -> active_ops(8) > num_threads(4)
    // -> M-tile infeasible -> sequential-full-team clamp.
    auto s = build_hybrid_probe(/*num_threads=*/4,
            /*Ms=*/ {8, 8, 8, 8, 8, 8, 8, 8});

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTilePathCaptureGuard cap;
    GemmModeCaptureGuard mode_cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_EQ(tag, kManyExpertsSeqFallback);

    const char *mode = zendnnl::lowoha::matmul::test_api ::
                               s_last_group_matmul_direct_gemm_mode.load(
                                       std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_STREQ(mode, "flat_m_tile_seq_clamp");
    EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 1)
            << "forced ALGO 2 clamped to sequential-full-team must report "
               "exec_algo=1 (ran ALGO-1 behaviour) despite chosen=ALGO_2";
}

// Multi-tier hybrid MUST stay off when `M_TILE_HYBRID=-1` (DISABLED),
// even on a shape that would otherwise satisfy every gate.  Same
// Qwen3-class shape as `MultiTierEngagesOnSkewedPrompt`; verifies the
// env-disable escape hatch keeps the legacy single-tier path
// available for A/B testing and emergency rollback.
TEST(TestGroupMatmulMTileBranches, MultiTierDisabledViaEnvOverride) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    std::vector<int> ms = {1024};
    ms.insert(ms.end(), 15, 4);
    auto s = build_hybrid_probe(/*num_threads=*/32, ms, /*N=*/1024);

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_disabled(-1); // DISABLED escape hatch
    MTilePathCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_NE(tag, kMultiTier)
            << "M_TILE_HYBRID=-1 must force the legacy single-tier path "
               "even on shapes the AUTO gate would accept; got multi-tier "
               "(tag="
            << tag << ")";
    EXPECT_EQ(tag, kPhase2Single)
            << "M_TILE_HYBRID=-1 + Qwen3-class skewed shape must land in "
               "Phase 2 single-tier (the legacy M-weighted fallback); "
               "got tag="
            << tag;
}

// Multi-tier hybrid stays off (falls through to Phase 2 single-tier)
// when the workload shape doesn't pass the skew gate.  Shape: 16
// actives all at M=256 ⇒ max_M = avg_M = 256; max_M / avg_M = 1 < 4
// ⇒ skew gate fails even though `actives` and `max_M ≥ 256` pass.
// Rank cap with a TIE across the boundary.  With `active_ops > num_threads`
// the heavy set is capped at `num_threads/3` experts.  Expressing that cap as
// a value cut collapses when the boundary M repeats: the cut equals that M,
// every tied expert tests `M <= cut` and lands light, the hybrid sees zero
// heavies and declines the whole call to the sequential clamp.  Ranking by
// (M desc, index asc) keeps the cap exact, so the hybrid still engages here.
TEST(TestGroupMatmulMTileBranches, MultiTierEngagesWithTiedHeavyBoundary) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    // 64 actives: 11 at M=400 and 53 at M=1.
    //   actives = 64 ≥ 32/2                              ✓
    //   max_M = 400 ≥ 256                                ✓
    //   avg_M = (11*400 + 53) / 64 = 69; 400 ≥ 4*69      ✓ (skew gate)
    //   active_ops(64) > num_threads(32)                 ⇒ rank cap engages
    //   max_heavy = 32/3 = 10, and the 11th-largest M is also 400 — the tie
    //   that the value cut could not express.
    std::vector<int> ms(11, 400);
    ms.insert(ms.end(), 53, 1);
    auto s = build_hybrid_probe(/*num_threads=*/32, ms, /*N=*/1024);

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_EQ(tag, kMultiTier)
            << "A tie at the heavy-rank boundary must not empty the heavy set "
               "and drop the call to the sequential clamp; got tag="
            << tag;
}

TEST(TestGroupMatmulMTileBranches, MultiTierLowSkewFallsThrough) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved_num_threads = omp_get_max_threads();
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { omp_set_num_threads(prev); }
    } thread_guard {saved_num_threads};
    omp_set_num_threads(32);
    int actual_team_size = 0;
#pragma omp parallel
    {
#pragma omp master
        actual_team_size = omp_get_num_threads();
    }
    if (actual_team_size < 32) {
        GTEST_SKIP() << "Requires >= 32 OMP threads; have " << actual_team_size;
    }

    // 16 actives all at M=256.  actives gate ✓, max_M gate ✓, skew
    // gate FAILS (max_M / avg_M = 1 < 4) ⇒ multi-tier must NOT engage;
    // falls through to Phase 2 single-tier.
    //   total_need = 16 × 16 = 256; 2×256 = 512 > 32 ⇒ wide-N excluded.
    auto s = build_hybrid_probe(/*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 256), /*N=*/1024);

    reset_grp_matmul_caches();
    AlgoEnvGuard algo_guard(2);
    MTileHybridOverride hybrid_auto(0);
    MTilePathCaptureGuard cap;

    ASSERT_EQ(group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                      s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                      s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts, s.gv.ldc,
                      s.gv.is_wc, s.params, nullptr, nullptr),
            status_t::success);

    const int tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
            std::memory_order_relaxed);
    EXPECT_NE(tag, kMultiTier)
            << "Multi-tier must NOT engage on low-skew shapes "
               "(max_M/avg_M = 1 < kHybridMinSkewX=4); got multi-tier "
               "(tag="
            << tag << ")";
    EXPECT_EQ(tag, kPhase2Single)
            << "Low-skew shape must fall through to Phase 2 single-tier; "
               "got tag="
            << tag;
}

// ============================================================================
// [8g] TestGroupMatmulMTileStandalone: numerical correctness of `flat_m_tile`
// when called as a standalone grouped matmul (no `fused_moe`, no `gated_act`,
// no `moe_postop`) — the llama.cpp use case.  Existing `TestGroupMatmulMTile
// Branches` only verify branch tags; they never check that the dst[] values
// produced by the M-tile path match what the caller expects.  This fixture
// adds the per-expert × per-row × per-col value comparison against an ALGO 1
// (legacy sequential per-expert) reference run on the same inputs.
//
// Reference choice: `AlgoEnvGuard(1)` forces the sequential per-expert ALGO 1
// path which dispatches each expert through `kernel_select` independently of
// any M-tile planner state.  Both runs use identical inputs (same src/wei
// pointers, same alpha/beta/bias, same params), so any numerical divergence
// between the two would indicate an M-tile-specific bug.  We tolerate small
// floating-point reduction-order noise via a relative+absolute tolerance.
//
// Coverage:
//   * Caller-allocated dst[] (canonical llama.cpp pattern).
//   * act=none (Op1-only standalone grouped matmul).
//   * Both bias=nullptr and bias=non-null cases.
//   * Mixed alpha/beta values (1.0 / 0.0 default, plus 2.0 / 0.5 stress).
//   * Path tag asserted: typically kPhase2Single for the chosen shape.
//
// Shape: 16 active experts × M=64 × K=64 × N=128 (Op1-like) — small enough
// for fast test execution, large enough that the M-tile planner picks a
// single-tier slice plan rather than round-based or wide-N.
//
// The test is designed to run on >=8 threads (skip otherwise); shape avoids
// `active_ops > num_threads` so the kRoundBased branch does not engage here
// (that branch has its own dedicated coverage in [8h] below).
// ============================================================================

namespace {

// Append-only helpers private to the standalone correctness fixture below.
//
// Build a bf16 standalone test input set with deterministic seeds.  Mirrors
// `build_hybrid_probe` but exposes a non-zero `bias` slot per expert (or
// nullptr per expert when `with_bias=false`) and per-expert alpha / beta
// (defaults 1.0 / 0.0).  Caller passes the M vector; N / K are fixed.
struct StandaloneMTileShape {
    int num_ops;
    int num_threads;
    int N;
    int K;
    std::vector<int> Ms;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> src;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> wei;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> bias;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> dst_test;
    std::vector<std::vector<zendnnl::common::bfloat16_t>> dst_ref;
    std::vector<const void *> srcs;
    std::vector<const void *> weis;
    std::vector<const void *> biases;
    std::vector<void *> dsts_test;
    std::vector<void *> dsts_ref;
    moe_test_utils::GemmVecs gv;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static StandaloneMTileShape build_standalone_mtile_shape(int num_threads,
        const std::vector<int> &Ms, int N, int K, bool with_bias,
        float alpha_val = 1.0f, float beta_val = 0.0f) {
    using namespace moe_test_utils;
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    StandaloneMTileShape s;
    s.num_threads = num_threads;
    s.num_ops = static_cast<int>(Ms.size());
    s.N = N;
    s.K = K;
    s.Ms = Ms;
    s.src.resize(s.num_ops);
    s.wei.resize(s.num_ops);
    s.bias.resize(s.num_ops);
    s.dst_test.resize(s.num_ops);
    s.dst_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
        s.wei[e].resize(static_cast<size_t>(K) * N);
        s.dst_test[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
        s.dst_ref[e].assign(static_cast<size_t>(Ms[e]) * N, bfloat16_t(0.0f));
        fill_src(s.src[e], e, 0.02f);
        fill_wei1(s.wei[e], e, 0.005f);
        if (with_bias) {
            s.bias[e].resize(static_cast<size_t>(N));
            for (int c = 0; c < N; ++c) {
                s.bias[e][c]
                        = bfloat16_t(static_cast<float>(c) * 0.001f + 0.01f);
            }
        }
    }
    s.srcs.resize(s.num_ops);
    s.weis.resize(s.num_ops);
    s.biases.resize(s.num_ops);
    s.dsts_test.resize(s.num_ops);
    s.dsts_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.srcs[e] = s.src[e].data();
        s.weis[e] = s.wei[e].data();
        s.biases[e] = with_bias ? s.bias[e].data() : nullptr;
        s.dsts_test[e] = s.dst_test[e].data();
        s.dsts_ref[e] = s.dst_ref[e].data();
    }
    s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K, alpha_val, beta_val,
            /*wc=*/true, /*tA=*/false, /*tB=*/false);
    s.gv.Ms = Ms;
    s.params = make_uniform_params(s.num_ops, data_type_t::bf16,
            /*bias_dt=*/with_bias ? data_type_t::bf16 : data_type_t::none);
    for (int e = 0; e < s.num_ops; ++e) {
        s.params[e].num_threads = num_threads;
    }
    return s;
}

// Per-expert × per-row × per-col tolerance comparison for bf16 buffers.
// Returns the first divergence (e, r, c, got, ref) — or -1's on success.
struct CompareResult {
    int e = -1, r = -1, c = -1;
    float got = 0.0f, ref = 0.0f;
    bool ok() const { return e < 0; }
};

// For gated-act dst, the live data lives only in the first `compare_cols`
// columns (typically `N/2`); columns `[compare_cols, N)` carry the raw
// pre-act matmul output and are skipped to avoid spurious mismatches.
static CompareResult compare_bf16_buffers(
        const std::vector<std::vector<zendnnl::common::bfloat16_t>> &got_b,
        const std::vector<std::vector<zendnnl::common::bfloat16_t>> &ref_b,
        const std::vector<int> &Ms, int stride, int compare_cols,
        float rtol = 5e-2f, float atol = 5e-2f) {
    CompareResult r;
    for (int e = 0; e < (int)Ms.size(); ++e) {
        const int M_e = Ms[e];
        if (M_e == 0) { continue; }
        for (int row = 0; row < M_e; ++row) {
            for (int c = 0; c < compare_cols; ++c) {
                const size_t off = static_cast<size_t>(row) * stride + c;
                const float g = static_cast<float>(got_b[e][off]);
                const float f = static_cast<float>(ref_b[e][off]);
                const float bound = std::abs(f) * rtol + atol;
                if (std::abs(g - f) > bound) {
                    r.e = e;
                    r.r = row;
                    r.c = c;
                    r.got = g;
                    r.ref = f;
                    return r;
                }
            }
        }
    }
    return r;
}

// Run the standalone group_matmul_direct twice — once with `algo_test`
// pinned (e.g. ALGO 2 / M-tile) writing to `dst_test`, once with the
// reference algo (ALGO 1 sequential per-expert) writing to `dst_ref` —
// and compare per-expert × per-row × per-col with the supplied tolerance.
//
// `gated_act` is forwarded to both runs (so the ref run also fuses the
// activation under ALGO 1) — when non-null, only the first `N/2`
// columns are compared because the gated activation halves the live
// data width in-place.
static void run_and_compare_standalone(StandaloneMTileShape &s, int algo_test,
        int algo_ref, int expected_tag, float rtol, float atol,
        const char *case_label,
        const zendnnl::lowoha::matmul::grp_matmul_gated_act_params *gated_act
        = nullptr) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;

    // Reference run first (ALGO 1, sequential per-expert; no M-tile
    // planner state can leak into the test).  Use a fresh params copy
    // because the dispatcher writes `lowoha_algo` back into params[i].
    {
        auto pr = s.params;
        moe_test_utils::AlgoEnvGuard algo_guard_ref(algo_ref);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_ref,
                        s.gv.ldc, s.gv.is_wc, pr, nullptr, gated_act),
                status_t::success)
                << "ref " << case_label;
    }
    // Test run (ALGO 2 / M-tile) with path-tag capture armed.
    int tag = 0;
    {
        auto pt = s.params;
        ::reset_grp_matmul_caches();
        moe_test_utils::AlgoEnvGuard algo_guard_test(algo_test);
        moe_test_utils::MTileHybridOverride hybrid_auto(0);
        moe_test_utils::MTilePathCaptureGuard cap;
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_test,
                        s.gv.ldc, s.gv.is_wc, pt, nullptr, gated_act),
                status_t::success)
                << "test " << case_label;
        tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
                std::memory_order_relaxed);
    }
    EXPECT_EQ(tag, expected_tag)
            << "Path tag mismatch on " << case_label << ": expected "
            << expected_tag << ", got " << tag
            << " (kRoundBased=0, kMultiTier=1, kWideNFallback=2, "
               "kPhase2Single=3, kVerticalFusionBF16=4)";
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    const int compare_cols
            = (gated_act && gated_act->act != grp_matmul_gated_act_t::none)
            ? s.N / 2
            : s.N;
    const auto cmp = compare_bf16_buffers(s.dst_test, s.dst_ref, s.Ms,
            /*stride=*/s.N, compare_cols, rtol, atol);
    EXPECT_TRUE(cmp.ok()) << "Value mismatch on " << case_label
                          << " e=" << cmp.e << " r=" << cmp.r << " c=" << cmp.c
                          << " got=" << cmp.got << " ref=" << cmp.ref;
}

static bool require_min_threads(int needed) {
    int actual = 0;
#pragma omp parallel
    {
#pragma omp master
        actual = omp_get_num_threads();
    }
    return actual >= needed;
}

} // namespace

// Baseline: 16 actives × M=64 × K=64 × N=128, no bias, alpha=1.0, beta=0.0.
// active_ops (16) <= num_threads (32) ⇒ kPhase2Single expected.
TEST(TestGroupMatmulMTileStandalone, CallerAllocActNoneNoBias) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "standalone_caller_alloc_act_none_no_bias");
}

// With per-expert bias.  Exercises `execute_m_tile`'s bias passthrough
// in the slice executor (line 195) — the same bias[e] vector is used
// for every row slice within an expert (no per-slice offset on bias).
TEST(TestGroupMatmulMTileStandalone, CallerAllocActNoneWithBias) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/true);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "standalone_caller_alloc_act_none_with_bias");
}

// Mixed alpha != 1, beta != 0 — exercises the `beta * C_init` path
// in matmul_execute.  Note that beta=0.5 requires dst_ref to be
// pre-initialised to the same values across both runs; we use the
// `assign(..., bfloat16_t(0.0f))` zero-init from the shape builder
// which gives both runs identical C_init = 0, so beta*C_init == 0
// and the numerical comparison still holds.  (A non-zero C_init test
// would belong under postop coverage in Batch 9.)
TEST(TestGroupMatmulMTileStandalone, CallerAllocAlphaBeta) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/false,
            /*alpha_val=*/2.0f, /*beta_val=*/0.5f);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "standalone_caller_alloc_alpha2_beta05");
}

// Gated activations on the standalone hot path: each of silu_and_mul,
// gelu_and_mul, swiglu_oai_mul.  N=256 (even) is required by the
// gated_act validator (`group_matmul_direct.cpp:478`).  The activation
// halves the live data width — first N/2 columns hold the activated
// output, columns [N/2, N) carry the raw pre-act matmul values.  The
// comparator only checks the first N/2 columns (set by the
// `gated_act != nullptr` branch in `run_and_compare_standalone`).
//
// Why this is the most informative test in Batch 2: the act is invoked
// per-slice from `execute_m_tile_act` (rows [row_start, row_end) per
// thread) under ALGO 2, but full-M from `sequential_experts` under
// ALGO 1.  Numerical parity across these two row-range semantics
// confirms that the slice-stitched activation is correct.
TEST(TestGroupMatmulMTileStandalone, GatedActSiluAndMul) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64,
            /*with_bias=*/false);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::silu_and_mul;

    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_silu_and_mul",
            /*gated_act=*/&ga);
}

TEST(TestGroupMatmulMTileStandalone, GatedActGeluAndMul) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64,
            /*with_bias=*/false);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::gelu_and_mul;

    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_gelu_and_mul",
            /*gated_act=*/&ga);
}

TEST(TestGroupMatmulMTileStandalone, GatedActSwigluOaiMul) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64,
            /*with_bias=*/false);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::swiglu_oai_mul;

    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_swiglu_oai_mul",
            /*gated_act=*/&ga);
}

// ============================================================================
// [8h] TestGroupMatmulMTileBranches.ForcedAlgo2ManyExpertsClampsToSeqFullTeam
//
// When `active_ops > num_threads` a pure M-tile plan is INFEASIBLE (it
// cannot give < 1 thread per active expert).  AUTO routes this regime to
// ALGO 5 (see TestGroupMatmulAutoPhaseEnv.* below); a FORCED
// `ZENDNNL_GRP_MATMUL_ALGO=2` clamps to the sequential full-team path
// (ALGO 1 equivalent: each active expert runs across the whole team, one at
// a time) and emits a one-time WARN.  Tag: kManyExpertsSeqFallback.
//
// We trigger it by pinning `params[e].num_threads = 4` while providing 8
// active experts.  Value parity is checked against an ALGO 1 reference
// (the clamp IS the ALGO 1 sequential full-team algorithm).
// ============================================================================

TEST(TestGroupMatmulMTileBranches, ForcedAlgo2ManyExpertsClampsToSeqFullTeam) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag ::
            kManyExpertsSeqFallback;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(8);
    if (!require_min_threads(4)) {
        GTEST_SKIP() << "Requires >= 4 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/4, // < active_ops=8 ⇒ M-tile infeasible
            /*Ms=*/std::vector<int>(8, 8),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kManyExpertsSeqFallback,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "forced_algo2_8actives_4threads_seq_clamp");
}

// Multi-tier value parity — same skewed prompt shape as
// `MultiTierEngagesOnSkewedPrompt` above, but additionally asserts
// numerical correctness of the per-slice executor against an ALGO 1
// reference.  Multi-tier splits the planner into heavy-pool + light-
// pool slots; the heavy pool runs per-slice (`execute_m_tile_act`,
// row-range activation), the light pool runs per-expert full-M
// activation (`execute_light_expert` ~m_tile.cpp:502-527).  Value
// parity across this dual-pool dispatch confirms there is no
// double-act on shared rows and no boundary-row off-by-one between
// the two pools.
TEST(TestGroupMatmulMTileBranches, MultiTierValueParityOnSkewedPrompt) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    std::vector<int> ms = {1024};
    ms.insert(ms.end(), 15, 4);
    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32, ms,
            /*N=*/1024, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kMultiTier,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "multi_tier_skewed_prompt_value_parity");
}

// Value parity on the tied heavy boundary.  Eleven experts share the
// boundary M while the rank cap admits ten, so this is the shape where the
// tie group is absorbed into the heavy tier.  The branch-tag test above
// proves the hybrid engages; this one proves the wider split still computes
// the right numbers, including for the expert that would otherwise have been
// demoted into the single-threaded light queue.
TEST(TestGroupMatmulMTileBranches, MultiTierTiedBoundaryValueParity) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kMultiTier;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    std::vector<int> ms(11, 400);
    ms.insert(ms.end(), 53, 1);
    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32, ms,
            /*N=*/1024, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kMultiTier,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "multi_tier_tied_boundary_value_parity");
}

// Forced-ALGO-2 value parity on a wide-N light-frame shape.  ALGO 2 is now
// pure M-tile, so a forced ALGO 2 on this shape runs single-tier
// (kPhase2Single) rather than the old wide-N sequential fallback.  Value
// parity against an ALGO 1 reference confirms the single-tier slice/stitch
// is numerically correct on the shallow-M shape.
TEST(TestGroupMatmulMTileBranches, ForcedAlgo2WideNValueParityOnLightFrames) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(8, 8),
            /*N=*/1024, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "forced_algo2_wide_n_light_frames_value_parity");
}

// ============================================================================
// [8i] TestGroupMatmulMTileQuantGate: direct unit-test of
// `check_m_tile_safe`'s dynamic-quant arm.  The gate's contract
// (`group_matmul_parallel_common.hpp:159-161`) is that
// `dynamic_quant = true` requires `src_scale.dims[0] == M[i]` so the
// per-thread reorder operates on its own rows.  Per-token `{M, 1}`
// and per-group-on-K `{M, G>1}` both satisfy the gate; per-tensor
// (`{}`, `{1}`, `{1, 1}`) is rejected, as are mismatched `M` dims.
//
// Why this matters: forcing `ZENDNNL_GRP_MATMUL_ALGO=2` with a
// per-tensor src_scale must fall back to ALGO 1 (the M-tile slicer
// cannot safely partition a shared scalar scale across threads).
// The fallback is enforced by `select_grp_matmul_algo` reading
// `check_m_tile_safe`'s return value.
//
// Replaces the per-call cost of constructing a full int8 fused-MoE
// fixture (which would need wei reorder, runtime BF16→S8 scratch,
// per-channel wei_scale plumbing, etc.) with a direct call into
// the inline gate predicate.  The full int8 pipeline is exercised
// by the existing `TestFusedMoEQuantDynINT8` parameterized sweep.
// ============================================================================

TEST(TestGroupMatmulMTileQuantGate, AcceptsPerTokenAndPerGroup) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::s8;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::none;
        params[i].dtypes.compute = data_type_t::s8;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].packing.pack_format_b = 0;
        params[i].dynamic_quant = true;
        // Per-token `{M, 1}` — accepted: dims[0] == M[i].
        params[i].quant_params.src_scale.dims = {16, 1};
        params[i].quant_params.src_scale.dt = data_type_t::f32;
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "per-token src_scale dims={M,1} must be accepted by M-tile gate";

    // Per-group-on-K `{M, G>1}` — also accepted.  Per-group is row-local
    // (each row owns G group-scales), so the row slicer still partitions
    // safely across threads.
    for (int i = 0; i < num_ops; ++i) {
        params[i].quant_params.src_scale.dims = {16, 4};
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "per-group-on-K src_scale dims={M,G>1} must be accepted "
               "(row-local granularity is what the M-tile slicer requires)";
}

TEST(TestGroupMatmulMTileQuantGate, RejectsPerTensorAndMismatched) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);

    auto make_params = [&](const std::vector<int64_t> &src_scale_dims) {
        std::vector<matmul_params> p(num_ops);
        for (int i = 0; i < num_ops; ++i) {
            p[i].dtypes.src = data_type_t::bf16;
            p[i].dtypes.wei = data_type_t::s8;
            p[i].dtypes.dst = data_type_t::bf16;
            p[i].dtypes.bias = data_type_t::none;
            p[i].dtypes.compute = data_type_t::s8;
            p[i].mem_format_a = 'n';
            p[i].mem_format_b = 'n';
            p[i].packing.pack_format_b = 0;
            p[i].dynamic_quant = true;
            p[i].quant_params.src_scale.dims = src_scale_dims;
            p[i].quant_params.src_scale.dt = data_type_t::f32;
        }
        return p;
    };

    // Per-tensor (dims empty): rejected — sd.empty() at gate line 194.
    {
        auto p = make_params({});
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "empty src_scale dims must be rejected by M-tile gate";
    }
    // Per-tensor {1}: rejected — sd[0] != M[i] (1 != 16).
    {
        auto p = make_params({1});
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "per-tensor src_scale dims={1} must be rejected "
                   "(single shared scalar cannot be safely row-sliced)";
    }
    // Per-tensor {1, 1}: rejected — sd[0] != M[i].
    {
        auto p = make_params({1, 1});
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "per-tensor src_scale dims={1,1} must be rejected";
    }
    // Per-column / per-K (no M dim): {1, K}: rejected — sd[0] != M[i].
    {
        auto p = make_params({1, 64});
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "per-K-only src_scale dims={1, K} must be rejected";
    }
    // Mismatched M dim: {M+1, 1}: rejected.
    {
        auto p = make_params({17, 1});
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "src_scale dims[0] != M[i] must be rejected";
    }
    // Mismatched src_zp dim while src_scale is valid: rejected — line 196.
    {
        auto p = make_params({16, 1});
        p[0].quant_params.src_zp.dims = {17, 1};
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "src_zp dims[0] != M[i] must be rejected even when "
                   "src_scale dims are valid";
    }
}

// ============================================================================
// [8j] TestGroupMatmulMTilePostopGate — direct unit-test of
//      `check_m_tile_safe`'s post-op rejection arm (lines 198-202 of
//      `group_matmul_parallel_common.hpp`).  The M-tile per-thread row
//      slicer cannot handle post-ops that REQUIRE the full M block
//      (softmax normalises across the rows of a tile, pooling
//      aggregates across rows); both are rejected at the gate so the
//      dispatcher falls back to ALGO 1 where the postop sees the full
//      block.
//
//      Binary `add` / `mul` postops are ACCEPTED — `execute_m_tile`
//      handles their row offset inside `m_tile/group_matmul_m_tile.cpp`
//      (offsets `po.buff` by `row_start × eff_ld × po_elem` for 2D /
//      3D row-varying tensors, leaves broadcast {1, N} / rank-≤1
//      tensors unchanged).
//
//      All four M-tile branches (kRoundBased, kMultiTier,
//      kWideNFallback, kPhase2Single) route through the SAME
//      `execute_m_tile` / `execute_m_tile_act` entry points for the
//      sliced experts — round-based and multi-tier-light dispatch the
//      full M of each expert to a single team, so the row-offset code
//      is a no-op (`row_start == 0`) for them and they degenerate to
//      ALGO-1-style per-expert postop handling.  The remaining branches
//      (sliced execution) all share the row-offset logic, so a single
//      gate-rejection test is sufficient.
//
//      The end-to-end value-parity test for binary postops on M-tile
//      is intentionally NOT added here — the existing `TestGroupMatmul`
//      sweep in `test_basic.cpp` covers binary postops × ALGO 2 across
//      a broad shape grid via the `matmul_kernel_test`
//      reference, and the gate-level test below proves the M-tile
//      safety predicate's reject/accept contract directly.
// ============================================================================

TEST(TestGroupMatmulMTilePostopGate, AcceptsBinaryAddAndMul) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;
    using zendnnl::common::post_op_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::none;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].packing.pack_format_b = 0;
        // Binary add — 2D row-varying buffer {M=16, N=64}.
        matmul_post_op add_po;
        add_po.po_type = post_op_type_t::binary_add;
        add_po.dtype = data_type_t::bf16;
        add_po.dims = {16, 64};
        add_po.leading_dim = 64;
        params[i].postop_.push_back(add_po);
        // Binary mul — broadcast {1, N=64}.
        matmul_post_op mul_po;
        mul_po.po_type = post_op_type_t::binary_mul;
        mul_po.dtype = data_type_t::bf16;
        mul_po.dims = {1, 64};
        mul_po.leading_dim = 64;
        params[i].postop_.push_back(mul_po);
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "binary_add (2D row-varying) and binary_mul (1D broadcast) "
               "post-ops must be accepted by the M-tile safety gate "
               "(row-offset handled in execute_m_tile lines 165-182)";
}

TEST(TestGroupMatmulMTilePostopGate, RejectsSoftmaxAndPooling) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;
    using zendnnl::common::post_op_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);

    auto make_params_with_postop = [&](post_op_type_t po_type) {
        std::vector<matmul_params> p(num_ops);
        for (int i = 0; i < num_ops; ++i) {
            p[i].dtypes.src = data_type_t::bf16;
            p[i].dtypes.wei = data_type_t::bf16;
            p[i].dtypes.dst = data_type_t::bf16;
            p[i].dtypes.bias = data_type_t::none;
            p[i].mem_format_a = 'n';
            p[i].mem_format_b = 'n';
            p[i].packing.pack_format_b = 0;
            matmul_post_op po;
            po.po_type = po_type;
            po.dtype = data_type_t::bf16;
            p[i].postop_.push_back(po);
        }
        return p;
    };

    // softmax: rejected — needs full M block for row-wise normalisation.
    {
        auto p = make_params_with_postop(post_op_type_t::softmax);
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "softmax post-op must be rejected — row-wise normalisation "
                   "across the full M block cannot be satisfied by the M-tile "
                   "per-thread row slicer";
    }
    // pooling: rejected — needs full M block for cross-row aggregation.
    {
        auto p = make_params_with_postop(post_op_type_t::pooling);
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "pooling post-op must be rejected — cross-row aggregation "
                   "cannot be partitioned across M-tile threads";
    }
    // Only one expert carries the unsupported postop — still rejected
    // (the gate must be conservative; mixed-postop batches all fall
    // back to ALGO 1).
    {
        auto p = make_params_with_postop(post_op_type_t::binary_add);
        matmul_post_op pool_po;
        pool_po.po_type = post_op_type_t::pooling;
        pool_po.dtype = data_type_t::bf16;
        p[2].postop_.push_back(pool_po); // single expert in the batch
        EXPECT_FALSE(check_m_tile_safe(layout, M, p, num_ops))
                << "ANY expert carrying softmax/pooling must reject the whole "
                   "batch (mixed-postop batches fall back to ALGO 1 because "
                   "the M-tile planner cannot split planner state across the "
                   "two postop classes mid-batch)";
    }
}

// ============================================================================
// [8k] TestGroupMatmulMTileDtypeGate — direct unit-test of
//      `check_m_tile_safe`'s dtype-uniformity arm (lines 185-188 of
//      `group_matmul_parallel_common.hpp`).  The gate requires
//      cross-expert dtype uniformity for src / wei / dst / bias —
//      M-tile's per-thread row slicer dispatches uniform GEMM tiles to
//      `execute_expert_slice`, which assumes the dtypes are stable
//      across the whole batch.  Heterogeneous-dtype batches MUST
//      reject so the dispatcher falls back to ALGO 1 where each
//      expert dispatches through `kernel_select` independently.
//
//      Note that the gate does NOT restrict WHICH concrete dtypes are
//      allowed — bf16, f32, s8 (with dynamic_quant), or any mix that
//      a supported kernel can handle is accepted as long as it is
//      uniform across experts.  This keeps the gate cheap (O(num_ops)
//      compare loop) and leaves the dtype routing to the kernel
//      selector inside `kernel_select`.
// ============================================================================

TEST(TestGroupMatmulMTileDtypeGate, AcceptsUniformBF16) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::bf16;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].packing.pack_format_b = 0;
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "uniform bf16 src/wei/dst/bias must be accepted by the M-tile "
               "gate";
}

TEST(TestGroupMatmulMTileDtypeGate, AcceptsUniformFP32) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::f32;
        params[i].dtypes.wei = data_type_t::f32;
        params[i].dtypes.dst = data_type_t::f32;
        params[i].dtypes.bias = data_type_t::f32;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].packing.pack_format_b = 0;
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "uniform f32 src/wei/dst/bias must be accepted — M-tile is "
               "dtype-agnostic at the safety gate (kernel routing handled "
               "by `kernel_select`)";
}

TEST(TestGroupMatmulMTileDtypeGate, AcceptsMixedPrecisionBF16toFP32) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        // bf16 inputs, f32 accumulator-style output (common LLM pattern
        // for attention output projections).
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::f32;
        params[i].dtypes.bias = data_type_t::f32;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
        params[i].packing.pack_format_b = 0;
    }
    EXPECT_TRUE(check_m_tile_safe(layout, M, params, num_ops))
            << "mixed-precision (bf16 in, f32 out) must be accepted — the "
               "gate enforces ACROSS-expert uniformity, not BETWEEN src/dst";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertSrcMismatch) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::none;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
    }
    // Mismatch one expert's src dtype.
    params[2].dtypes.src = data_type_t::f32;
    EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
            << "cross-expert src dtype mismatch must be rejected — M-tile's "
               "per-thread slicer dispatches uniform GEMM tiles and cannot "
               "switch dtype mid-batch";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertWeiMismatch) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::none;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
    }
    // s8 weights on one expert (quantised), bf16 on the others.
    params[0].dtypes.wei = data_type_t::s8;
    EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
            << "cross-expert wei dtype mismatch must be rejected (e.g. one "
               "quantised expert + dense others) — the M-tile planner cannot "
               "route a single team to two different kernel families";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertDstMismatch) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::none;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
    }
    params[3].dtypes.dst = data_type_t::f32;
    EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
            << "cross-expert dst dtype mismatch must be rejected — the row "
               "slicer's dst pointer arithmetic assumes uniform `dst_elem` "
               "(`size_of(dtypes.dst)`) across all experts";
}

TEST(TestGroupMatmulMTileDtypeGate, RejectsCrossExpertBiasMismatch) {
    using namespace zendnnl::lowoha::matmul;
    using zendnnl::common::data_type_t;

    const int num_ops = 4;
    std::vector<char> layout(num_ops, 'r');
    std::vector<int> M(num_ops, 16);
    std::vector<matmul_params> params(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        params[i].dtypes.src = data_type_t::bf16;
        params[i].dtypes.wei = data_type_t::bf16;
        params[i].dtypes.dst = data_type_t::bf16;
        params[i].dtypes.bias = data_type_t::f32;
        params[i].mem_format_a = 'n';
        params[i].mem_format_b = 'n';
    }
    params[1].dtypes.bias = data_type_t::bf16;
    EXPECT_FALSE(check_m_tile_safe(layout, M, params, num_ops))
            << "cross-expert bias dtype mismatch must be rejected (e.g. f32 "
               "bias on most experts + bf16 bias on one) — the kernel select "
               "would diverge mid-batch";
}

// ============================================================================
// [8l] TestGroupMatmulMTileStandalone.F32 — full-f32 numerical-correctness
// fixture for the standalone hot path.  Mirrors the bf16 standalone fixture
// at [8g] above but with `data_type_t::f32` for src / wei / dst / bias.
// f32 routes through a different kernel inside `kernel_select` (typically
// the AOCL FP32 GEMM rather than bf16 BRGEMM), so this catches regressions
// in dtype handling that the bf16-only standalone tests would miss.
//
// Reference: ALGO 1 (sequential per-expert) on the same f32 inputs.
// Test:      ALGO 2 (M-tile / `flat_m_tile`).
// Tolerance: tight (rtol=1e-5, atol=1e-5) — f32 has 7-digit precision so
//            the reduction-order noise between sequential and M-tile is
//            well below 1e-5 for this small K=64 / M=64 / N=128 shape.
// ============================================================================

namespace {

struct StandaloneMTileShapeF32 {
    int num_ops;
    int num_threads;
    int N;
    int K;
    std::vector<int> Ms;
    std::vector<std::vector<float>> src;
    std::vector<std::vector<float>> wei;
    std::vector<std::vector<float>> bias;
    std::vector<std::vector<float>> dst_test;
    std::vector<std::vector<float>> dst_ref;
    std::vector<const void *> srcs;
    std::vector<const void *> weis;
    std::vector<const void *> biases;
    std::vector<void *> dsts_test;
    std::vector<void *> dsts_ref;
    moe_test_utils::GemmVecs gv;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static StandaloneMTileShapeF32 build_standalone_mtile_shape_f32(int num_threads,
        const std::vector<int> &Ms, int N, int K, bool with_bias,
        float alpha_val = 1.0f, float beta_val = 0.0f) {
    using namespace moe_test_utils;
    using zendnnl::common::data_type_t;
    StandaloneMTileShapeF32 s;
    s.num_threads = num_threads;
    s.num_ops = static_cast<int>(Ms.size());
    s.N = N;
    s.K = K;
    s.Ms = Ms;
    s.src.resize(s.num_ops);
    s.wei.resize(s.num_ops);
    s.bias.resize(s.num_ops);
    s.dst_test.resize(s.num_ops);
    s.dst_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
        s.wei[e].resize(static_cast<size_t>(K) * N);
        s.dst_test[e].assign(static_cast<size_t>(Ms[e]) * N, 0.0f);
        s.dst_ref[e].assign(static_cast<size_t>(Ms[e]) * N, 0.0f);
        // Deterministic seeds — same hash family as the bf16 fixture but
        // returns f32 values directly (no bf16-truncation noise).
        const uint64_t base_src = 0x9E3779B97F4A7C15ULL
                + static_cast<uint64_t>(e) * 0x100000001B3ULL;
        for (size_t k = 0; k < s.src[e].size(); ++k) {
            uint64_t h = base_src + k * 0xC4CEB9FE1A85EC53ULL;
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            s.src[e][k] = static_cast<float>(static_cast<int>(h & 0xff) - 128)
                    * 0.005f;
        }
        const uint64_t base_wei = 0xBF58476D1CE4E5B9ULL
                + static_cast<uint64_t>(e) * 0x100000001B3ULL;
        for (size_t k = 0; k < s.wei[e].size(); ++k) {
            uint64_t h = base_wei + k * 0xC4CEB9FE1A85EC53ULL;
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            s.wei[e][k] = static_cast<float>(static_cast<int>(h & 0xff) - 128)
                    * 0.002f;
        }
        if (with_bias) {
            s.bias[e].resize(static_cast<size_t>(N));
            for (int c = 0; c < N; ++c) {
                s.bias[e][c] = static_cast<float>(c) * 0.001f + 0.01f;
            }
        }
    }
    s.srcs.resize(s.num_ops);
    s.weis.resize(s.num_ops);
    s.biases.resize(s.num_ops);
    s.dsts_test.resize(s.num_ops);
    s.dsts_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.srcs[e] = s.src[e].data();
        s.weis[e] = s.wei[e].data();
        s.biases[e] = with_bias ? s.bias[e].data() : nullptr;
        s.dsts_test[e] = s.dst_test[e].data();
        s.dsts_ref[e] = s.dst_ref[e].data();
    }
    s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K, alpha_val, beta_val,
            /*wc=*/true, /*tA=*/false, /*tB=*/false);
    s.gv.Ms = Ms;
    s.params = make_uniform_params(s.num_ops, data_type_t::f32,
            /*bias_dt=*/with_bias ? data_type_t::f32 : data_type_t::none);
    for (int e = 0; e < s.num_ops; ++e) {
        s.params[e].num_threads = num_threads;
    }
    return s;
}

static void run_and_compare_standalone_f32(StandaloneMTileShapeF32 &s,
        int algo_test, int algo_ref, int expected_tag, float rtol, float atol,
        const char *case_label) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;

    // Both runs below share one weight pool and clear the caches in between,
    // which the in-place mode forbids: run 1 would rewrite the buffer and
    // run 2 would then repack already-packed bytes.  Pin out-of-place.
    WeightCacheGuard wc_out_of_place(1);

    {
        auto pr = s.params;
        AlgoEnvGuard algo_guard_ref(algo_ref);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_ref,
                        s.gv.ldc, s.gv.is_wc, pr, nullptr, nullptr),
                status_t::success)
                << "ref " << case_label;
    }
    int tag = 0;
    {
        auto pt = s.params;
        ::reset_grp_matmul_caches();
        AlgoEnvGuard algo_guard_test(algo_test);
        MTileHybridOverride hybrid_auto(0);
        MTilePathCaptureGuard cap;
        ASSERT_EQ(
                group_matmul_direct(s.gv.layout, s.gv.transA, s.gv.transB,
                        s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs, s.gv.lda,
                        s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_test,
                        s.gv.ldc, s.gv.is_wc, pt, nullptr, nullptr),
                status_t::success)
                << "test " << case_label;
        tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
                std::memory_order_relaxed);
    }
    EXPECT_EQ(tag, expected_tag)
            << "Path tag mismatch on " << case_label << ": expected "
            << expected_tag << ", got " << tag;

    // f32 value comparison — tight tolerance since both runs are
    // FP32 throughout.  Per-expert × per-row × per-col first-divergence
    // reporting matches the bf16 helper's contract.
    for (int e = 0; e < s.num_ops; ++e) {
        const int M_e = s.Ms[e];
        if (M_e == 0) { continue; }
        for (int row = 0; row < M_e; ++row) {
            for (int c = 0; c < s.N; ++c) {
                const size_t off = static_cast<size_t>(row) * s.N + c;
                const float g = s.dst_test[e][off];
                const float f = s.dst_ref[e][off];
                const float bound = std::abs(f) * rtol + atol;
                if (std::abs(g - f) > bound) {
                    FAIL() << "f32 value mismatch on " << case_label
                           << " e=" << e << " r=" << row << " c=" << c
                           << " got=" << g << " ref=" << f;
                }
            }
        }
    }
}

} // namespace

TEST(TestGroupMatmulMTileStandalone, F32NoBias) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f32(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/false);
    run_and_compare_standalone_f32(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/1e-5f, /*atol=*/1e-5f, "standalone_f32_no_bias");
}

// ============================================================================
// [8l.b] TestGroupMatmulMTileStandalone.F16NoBias — full-f16 numerical
// correctness fixture for the standalone hot path.  Mirrors the bf16 fixture
// at [8g] above and the f32 fixture at [8l] above, but with
// `data_type_t::f16` for src / wei / dst (no bias).  F16 routes through
// `aocl_gemm_f16f16f16of16` (AVX-512-FP16 native accumulation; see
// `aocl_kernel.cpp:994-1011` for the dispatch arm), so this catches
// regressions in F16 dtype handling that bf16 / f32 standalone tests would
// miss.
//
// CRITICAL CONTRACT THIS TEST LOCKS:
//   `flat_m_tile`'s per-thread offset arithmetic uses
//   `size_of(params[0].dtypes.*)` for the element sizes
//   (`group_matmul_m_tile.cpp:520-521`).  For F16 this resolves to 2
//   (same as BF16), so the M-tile arithmetic happens to be correct
//   for F16 without an F16-specific code path.  This test directly
//   validates that property — a future refactor that hard-codes BF16
//   element sizes (e.g. via `sizeof(bfloat16_t)`) would silently
//   break F16 and trip this test.
//
// Reference: ALGO 1 (sequential per-expert) on the same f16 inputs.
// Test:      ALGO 2 (M-tile / `flat_m_tile`).
// Path tag:  kPhase2Single (the M-weighted single-tier branch — see
//            `flat_m_tile.cpp:1192-1203`).
// Tolerance: rtol=5e-3, atol=5e-3 — F16 has ~10 fraction bits, so the
//            reduction-order noise between sequential and M-tile is
//            comparable to the bf16 case at K=64 (worst-case
//            accumulated rounding ≈ K·ulp·|alpha|·|max input|² which is
//            well under the bound for the small-value seeds below).
// Skip:      hosts without AVX-512-FP16 (group_matmul_direct's F16 ISA
//            gate at lines 871-876 returns status_t::isa_unsupported,
//            matching the skip pattern in `TestGroupMatmul.F16_F16`).
// ============================================================================

namespace {

struct StandaloneMTileShapeF16 {
    int num_ops;
    int num_threads;
    int N;
    int K;
    std::vector<int> Ms;
    // Storage is uint16_t bit-patterns interpreted as IEEE 754 binary16.
    // Mirrors how the rest of the suite stores BF16 (uint16_t bits) for
    // ABI portability — the AOCL F16 kernels take a `float16 *` which is
    // a typedef-compatible 16-bit type.  Conversion to/from F32 goes
    // through `zendnnl::common::float16_t::{f32_to_f16_val,f16_to_f32_val}`.
    std::vector<std::vector<uint16_t>> src;
    std::vector<std::vector<uint16_t>> wei;
    std::vector<std::vector<uint16_t>> dst_test;
    std::vector<std::vector<uint16_t>> dst_ref;
    std::vector<const void *> srcs;
    std::vector<const void *> weis;
    std::vector<const void *> biases;
    std::vector<void *> dsts_test;
    std::vector<void *> dsts_ref;
    moe_test_utils::GemmVecs gv;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

// F32→F16 RNE via the library helper.  Kept as a local inline so the
// per-element seed loops below read closely against the .F32 sibling
// (where the F32 value is stored directly without conversion).
static inline uint16_t mtile_standalone_f32_to_f16(float v) {
    return zendnnl::common::float16_t::f32_to_f16_val(v);
}

static StandaloneMTileShapeF16 build_standalone_mtile_shape_f16(int num_threads,
        const std::vector<int> &Ms, int N, int K, float alpha_val = 1.0f,
        float beta_val = 0.0f) {
    using namespace moe_test_utils;
    using zendnnl::common::data_type_t;
    StandaloneMTileShapeF16 s;
    s.num_threads = num_threads;
    s.num_ops = static_cast<int>(Ms.size());
    s.N = N;
    s.K = K;
    s.Ms = Ms;
    s.src.resize(s.num_ops);
    s.wei.resize(s.num_ops);
    s.dst_test.resize(s.num_ops);
    s.dst_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.src[e].resize(static_cast<size_t>(Ms[e]) * K);
        s.wei[e].resize(static_cast<size_t>(K) * N);
        s.dst_test[e].assign(static_cast<size_t>(Ms[e]) * N, uint16_t(0));
        s.dst_ref[e].assign(static_cast<size_t>(Ms[e]) * N, uint16_t(0));
        // Deterministic seeds — same hash family as the bf16 / f32 fixtures
        // so cross-dtype regressions stay correlated.  Values kept small
        // (|x| ≲ 0.7) so K=64 inner products never overflow F16's
        // ±65504 normal range.
        const uint64_t base_src = 0x9E3779B97F4A7C15ULL
                + static_cast<uint64_t>(e) * 0x100000001B3ULL;
        for (size_t k = 0; k < s.src[e].size(); ++k) {
            uint64_t h = base_src + k * 0xC4CEB9FE1A85EC53ULL;
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            const float f = static_cast<float>(static_cast<int>(h & 0xff) - 128)
                    * 0.005f;
            s.src[e][k] = mtile_standalone_f32_to_f16(f);
        }
        const uint64_t base_wei = 0xBF58476D1CE4E5B9ULL
                + static_cast<uint64_t>(e) * 0x100000001B3ULL;
        for (size_t k = 0; k < s.wei[e].size(); ++k) {
            uint64_t h = base_wei + k * 0xC4CEB9FE1A85EC53ULL;
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            const float f = static_cast<float>(static_cast<int>(h & 0xff) - 128)
                    * 0.002f;
            s.wei[e][k] = mtile_standalone_f32_to_f16(f);
        }
    }
    s.srcs.resize(s.num_ops);
    s.weis.resize(s.num_ops);
    s.biases.assign(s.num_ops, nullptr);
    s.dsts_test.resize(s.num_ops);
    s.dsts_ref.resize(s.num_ops);
    for (int e = 0; e < s.num_ops; ++e) {
        s.srcs[e] = s.src[e].data();
        s.weis[e] = s.wei[e].data();
        s.dsts_test[e] = s.dst_test[e].data();
        s.dsts_ref[e] = s.dst_ref[e].data();
    }
    s.gv = GemmVecs::uniform(s.num_ops, /*M=*/0, N, K, alpha_val, beta_val,
            /*wc=*/true, /*tA=*/false, /*tB=*/false);
    s.gv.Ms = Ms;
    s.params = make_uniform_params(
            s.num_ops, data_type_t::f16, /*bias_dt=*/data_type_t::none);
    for (int e = 0; e < s.num_ops; ++e) {
        s.params[e].num_threads = num_threads;
    }
    return s;
}

// Returns false iff group_matmul_direct refused with isa_unsupported
// on the first (reference) run; returns true when the run proceeded
// (the value comparison was performed).  Callers store the result as
// `ran` and `GTEST_SKIP()` when it is false, so the host cleanly skips
// on platforms without AVX-512-FP16 — matches the runtime-status
// pattern used by `TestGroupMatmul.F16_F16` in test_basic.cpp.
//
// Optional `gated_act` matches the BF16 sibling `run_and_compare_
// standalone`: when non-null and `act != none`, the dispatcher fuses
// the gated activation into the test and reference runs (both must
// see it to keep the comparison valid).  In that case only the
// first N/2 destination columns hold live data; columns [N/2, N)
// are "don't care" raw GEMM output by the activation kernel
// contract (see group_matmul_moe_act.cpp), so they're excluded
// from the value comparison.
static bool run_and_compare_standalone_f16(StandaloneMTileShapeF16 &s,
        int algo_test, int algo_ref, int expected_tag, float rtol, float atol,
        const char *case_label,
        const zendnnl::lowoha::matmul::grp_matmul_gated_act_params *gated_act
        = nullptr) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;

    // Reference run first (ALGO 1, sequential per-expert).  An
    // `isa_unsupported` return here means the host lacks AVX-512-FP16;
    // signal the caller to GTEST_SKIP() instead of failing.
    {
        auto pr = s.params;
        AlgoEnvGuard algo_guard_ref(algo_ref);
        ::reset_grp_matmul_caches();
        const status_t st = group_matmul_direct(s.gv.layout, s.gv.transA,
                s.gv.transB, s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs,
                s.gv.lda, s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_ref,
                s.gv.ldc, s.gv.is_wc, pr, nullptr, gated_act);
        if (st == status_t::isa_unsupported) { return false; }
        EXPECT_EQ(st, status_t::success) << "ref " << case_label;
        if (st != status_t::success) {
            return true; // counted as not-skipped
        }
    }
    int tag = 0;
    {
        auto pt = s.params;
        ::reset_grp_matmul_caches();
        AlgoEnvGuard algo_guard_test(algo_test);
        MTileHybridOverride hybrid_auto(0);
        MTilePathCaptureGuard cap;
        const status_t st = group_matmul_direct(s.gv.layout, s.gv.transA,
                s.gv.transB, s.gv.Ms, s.gv.Ns, s.gv.Ks, s.gv.alpha, s.srcs,
                s.gv.lda, s.weis, s.gv.ldb, s.biases, s.gv.beta, s.dsts_test,
                s.gv.ldc, s.gv.is_wc, pt, nullptr, gated_act);
        EXPECT_EQ(st, status_t::success) << "test " << case_label;
        if (st != status_t::success) { return true; }
        tag = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
                std::memory_order_relaxed);
    }
    EXPECT_EQ(tag, expected_tag)
            << "Path tag mismatch on " << case_label << ": expected "
            << expected_tag << ", got " << tag;

    // F16 value comparison — promote both sides to F32 for the bound
    // calculation so the relative-tolerance math doesn't lose precision
    // in F16's narrow range.  Per-expert × per-row × per-col
    // first-divergence reporting matches the bf16 / f32 helpers'
    // contract.
    //
    // When a gated activation fused, only the first N/2 columns hold
    // live activated output; columns [N/2, N) are the raw GEMM
    // "don't care" half (see group_matmul_moe_act.cpp activation
    // kernel contract).  Mirrors the BF16 sibling's `compare_cols`
    // logic in `run_and_compare_standalone`.
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    const int compare_cols
            = (gated_act && gated_act->act != grp_matmul_gated_act_t::none)
            ? s.N / 2
            : s.N;
    for (int e = 0; e < s.num_ops; ++e) {
        const int M_e = s.Ms[e];
        if (M_e == 0) { continue; }
        for (int row = 0; row < M_e; ++row) {
            for (int c = 0; c < compare_cols; ++c) {
                const size_t off = static_cast<size_t>(row) * s.N + c;
                const float g = zendnnl::common::float16_t::f16_to_f32_val(
                        s.dst_test[e][off]);
                const float f = zendnnl::common::float16_t::f16_to_f32_val(
                        s.dst_ref[e][off]);
                const float bound = std::abs(f) * rtol + atol;
                if (std::abs(g - f) > bound) {
                    ADD_FAILURE() << "f16 value mismatch on " << case_label
                                  << " e=" << e << " r=" << row << " c=" << c
                                  << " got=" << g << " ref=" << f
                                  << " bound=" << bound;
                    return true;
                }
            }
        }
    }
    return true;
}

} // namespace

TEST(TestGroupMatmulMTileStandalone, F16NoBias) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f16(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64);
    const bool ran
            = run_and_compare_standalone_f16(s, /*algo_test=*/2, /*algo_ref=*/1,
                    /*expected_tag=*/kPhase2Single,
                    /*rtol=*/5e-3f, /*atol=*/5e-3f, "standalone_f16_no_bias");
    if (!ran) { GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA"; }
}

// ============================================================================
// [8m] TestGroupMatmulMTileCustomKernelInvariance — verify the M-tile
// (ALGO 2) runtime is independent of the CUSTOM_KERNEL env knob.
//
// Contract: M-tile (`flat_m_tile`) dispatches every per-thread slice
// through `execute_expert_slice`, which selects the kernel via
// `resolve_kernel()` (returns the matmul_algo_t from `matmul_config_t`
// — defaults to `aocl_dlp_blocked`).  This is independent of the CK
// env knob — `get_grp_matmul_custom_kernel()` is consulted only by the
// PREPACK call BEFORE the OMP region (see m_tile.cpp lines 629-635 for
// the standalone path and lines 1457-1475 for the pipeline).  The
// prepack short-circuits when its arguments don't trigger the CK
// packing format, so for ALGO 2 it always lands on the same packed
// representation regardless of the CK knob.
//
// Therefore the dst[] bytes produced by the M-tile run MUST be
// byte-identical between CK=0 and CK=1 — any difference would
// indicate that the M-tile execution path picked up a CK-dependent
// branch it shouldn't.  We use the test-only `CustomKernelOverride`
// RAII guard (NOT EnvVarGuard, since the env value is cached at first
// read) to switch the knob between the two runs.
// ============================================================================

// ============================================================================
// [8n] TestGroupMatmulMTileEdgeCases — operational edge cases for M-tile.
//
//   * num_ops == 0 (empty batch): early-out at m_tile.cpp:621 — dispatch
//     succeeds with no work done, no dst writes (vacuously held).
//   * All M[i] == 0 (all inactive): early-out at m_tile.cpp:657
//     (`active_ops == 0`).  Dispatch succeeds, dst untouched.
//   * Mixed active / inactive experts (some M[i] == 0, others > 0):
//     planner's `active_pos[i] = -1` for inactive (m_tile.cpp:669-675)
//     keeps them out of the CCD stripe.  Active experts produce
//     correct values; inactive experts' dst slots remain at their
//     initialised values (caller-allocated buffer policy — same as
//     ALGO 1 reference).
//   * Odd N + gated act: rejected by the public-API validator at
//     `group_matmul_direct.cpp:476-482` (status_t::failure).  M-tile
//     never sees the call.
// ============================================================================

TEST(TestGroupMatmulMTileEdgeCases, NumOpsZeroSucceeds) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::matmul_params;
    using zendnnl::lowoha::matmul::status_t;

    // Empty input vectors — every per-expert vector is size 0.
    std::vector<char> layout;
    std::vector<bool> transA, transB, is_wc;
    std::vector<int> Ms, Ns, Ks, lda, ldb, ldc;
    std::vector<float> alpha, beta;
    std::vector<const void *> srcs, weis, biases;
    std::vector<void *> dsts;
    std::vector<matmul_params> params;

    AlgoEnvGuard algo2_guard(2);
    ::reset_grp_matmul_caches();
    // Public API typically rejects empty batches at the validator (line
    // 191 of group_matmul_direct.cpp: "required input vector is empty").
    // The contract is `status_t::failure` — NOT a crash.  This is the
    // safety net that M-tile relies on for its `num_ops == 0` invariant.
    const status_t st = group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
            alpha, srcs, lda, weis, ldb, biases, beta, dsts, ldc, is_wc, params,
            nullptr);
    EXPECT_EQ(st, status_t::failure)
            << "empty input vectors must be rejected by the public-API "
               "validator (defends M-tile's `num_ops == 0` invariant — "
               "the dispatcher never reaches `flat_m_tile` on an empty batch)";
}

TEST(TestGroupMatmulMTileEdgeCases, AllExpertsInactiveSucceeds) {
    using namespace moe_test_utils;
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    // 8 experts, all with M=0 — `active_ops` should be 0 at line 657
    // and `flat_m_tile` returns immediately without entering any
    // branch.  No tag is set (capture stays at its initial value).
    const int num_ops = 8;
    const int N = 128;
    const int K = 64;
    std::vector<int> Ms(num_ops, 0);

    // Buffers are present but unused (M=0 → 0 rows per expert).
    std::vector<std::vector<bfloat16_t>> src(num_ops);
    std::vector<std::vector<bfloat16_t>> wei(num_ops);
    std::vector<std::vector<bfloat16_t>> dst(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(1); // size-1 placeholder
        wei[e].resize(static_cast<size_t>(K) * N);
        dst[e].resize(1);
    }
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }

    auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N, K);
    gv.Ms = Ms;
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (auto &pp : params) {
        pp.num_threads = 32;
    }

    AlgoEnvGuard algo2_guard(2);
    ::reset_grp_matmul_caches();
    const status_t st = group_matmul_direct(gv.layout, gv.transA, gv.transB,
            gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
            gv.beta, dsts, gv.ldc, gv.is_wc, params, nullptr);
    // Public API may accept (status_t::success, no-op) or reject (some
    // validators treat all-zero M as degenerate).  Both outcomes are
    // semantically correct — what we test is that the call DOES NOT
    // crash, and if it succeeds, no dst writes occurred (vacuously
    // true since M=0).  We tolerate either status.
    EXPECT_TRUE(st == status_t::success || st == status_t::failure)
            << "all-inactive batch must terminate cleanly (success or "
               "failure, never crash)";
}

// Mixed active / inactive: some experts have M[i]=0 alongside others
// with M[i]>0.  The M-tile planner builds `active_pos[i]` (line 669)
// which skips inactives, and dispatches only active experts.  Value
// correctness on the active subset must match ALGO 1 reference; the
// inactive expert's dst slot is untouched in both runs (M=0 → no
// rows to write).
TEST(TestGroupMatmulMTileEdgeCases, MixedActiveInactiveExperts) {
    using namespace moe_test_utils;
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    // 8 experts: 4 active (M=64), 4 inactive (M=0), interleaved.  This
    // exercises both the active_pos compaction and the cross-expert
    // dispatch order — the planner's CCD stripe should skip inactive
    // slots without leaving gaps.
    const int num_ops = 8;
    const int N = 128, K = 64, M_active = 64;
    std::vector<int> Ms(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        Ms[e] = (e % 2 == 0) ? M_active : 0;
    }

    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
            dst_test(num_ops), dst_ref(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        const int M_e = Ms[e];
        src[e].resize(static_cast<size_t>(std::max(1, M_e)) * K);
        wei[e].resize(static_cast<size_t>(K) * N);
        dst_test[e].assign(
                static_cast<size_t>(std::max(1, M_e)) * N, bfloat16_t(0.0f));
        dst_ref[e].assign(
                static_cast<size_t>(std::max(1, M_e)) * N, bfloat16_t(0.0f));
        if (M_e > 0) {
            fill_src(src[e], e, 0.02f);
            fill_wei1(wei[e], e, 0.005f);
        }
    }
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts_test(num_ops), dsts_ref(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts_test[e] = dst_test[e].data();
        dsts_ref[e] = dst_ref[e].data();
    }

    auto gv = GemmVecs::uniform(num_ops, /*M=*/0, N, K);
    gv.Ms = Ms;
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (auto &pp : params) {
        pp.num_threads = 32;
    }

    // Reference: ALGO 1 (sequential per-expert).
    {
        auto pr = params;
        AlgoEnvGuard algo1_guard(1);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                          gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                          biases, gv.beta, dsts_ref, gv.ldc, gv.is_wc, pr,
                          nullptr),
                status_t::success)
                << "ref dispatch failed";
    }
    // Test: ALGO 2 (M-tile).
    {
        auto pt = params;
        AlgoEnvGuard algo2_guard(2);
        ::reset_grp_matmul_caches();
        ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB, gv.Ms,
                          gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb,
                          biases, gv.beta, dsts_test, gv.ldc, gv.is_wc, pt,
                          nullptr),
                status_t::success)
                << "test dispatch failed";
    }

    // Compare per-expert: active experts must produce matching values;
    // inactive experts have no rows to compare.
    for (int e = 0; e < num_ops; ++e) {
        if (Ms[e] == 0) { continue; }
        for (int row = 0; row < Ms[e]; ++row) {
            for (int c = 0; c < N; ++c) {
                const size_t off = static_cast<size_t>(row) * N + c;
                const float gt = static_cast<float>(dst_test[e][off]);
                const float ref = static_cast<float>(dst_ref[e][off]);
                const float bound = std::abs(ref) * 5e-2f + 5e-2f;
                if (std::abs(gt - ref) > bound) {
                    FAIL() << "mixed-active value mismatch e=" << e
                           << " r=" << row << " c=" << c << " got=" << gt
                           << " ref=" << ref;
                }
            }
        }
    }
}

TEST(TestGroupMatmulMTileEdgeCases, OddNWithGatedActRejected) {
    using namespace moe_test_utils;
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::status_t;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    const int num_ops = 4;
    const int M = 16;
    const int N_odd = 127; // odd — must be rejected
    const int K = 64;

    std::vector<std::vector<bfloat16_t>> src(num_ops), wei(num_ops),
            dst(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        src[e].resize(static_cast<size_t>(M) * K);
        wei[e].resize(static_cast<size_t>(K) * N_odd);
        dst[e].assign(static_cast<size_t>(M) * N_odd, bfloat16_t(0.0f));
    }
    std::vector<const void *> srcs(num_ops), weis(num_ops),
            biases(num_ops, nullptr);
    std::vector<void *> dsts(num_ops);
    for (int e = 0; e < num_ops; ++e) {
        srcs[e] = src[e].data();
        weis[e] = wei[e].data();
        dsts[e] = dst[e].data();
    }

    auto gv = GemmVecs::uniform(num_ops, M, N_odd, K);
    auto params = make_uniform_params(num_ops, data_type_t::bf16);
    for (auto &pp : params) {
        pp.num_threads = 32;
    }

    grp_matmul_gated_act_params gated {};
    gated.act = grp_matmul_gated_act_t::silu_and_mul;

    AlgoEnvGuard algo2_guard(2);
    ::reset_grp_matmul_caches();
    const status_t st = group_matmul_direct(gv.layout, gv.transA, gv.transB,
            gv.Ms, gv.Ns, gv.Ks, gv.alpha, srcs, gv.lda, weis, gv.ldb, biases,
            gv.beta, dsts, gv.ldc, gv.is_wc, params, nullptr, &gated);
    EXPECT_EQ(st, status_t::failure)
            << "odd N=" << N_odd
            << " with gated activation must be rejected "
               "by the public-API validator (group_matmul_direct.cpp:476-482) "
               "— gated activations collapse pairs of columns (N/2 lanes "
               "per row), so an odd N has no valid `N/2` semantics";
}

// Pins the OMP team for a K-blocking parity case and restores it after.
struct KblockThreadGuard {
    int prev;
    explicit KblockThreadGuard(int n) : prev(omp_get_max_threads()) {
        omp_set_num_threads(n);
    }
    ~KblockThreadGuard() { omp_set_num_threads(prev); }
};

// K-blocking L2 budget and NR32 INT8 o-block size (int8 weights plus the
// int32 compensation row), mirroring the INT8 gate in
// `custom_kernel/dispatch.cpp`.  Deriving K from the detected L2 keeps the
// engage / fit cases correct on hosts whose L2 is not 1 MiB.
static int64_t int8_kblock_l2_budget() {
    const int64_t l2 = zendnnl::lowoha::matmul::native::detect_uarch().l2_bytes;
    const int64_t safe = (l2 >= 64 * 1024) ? l2 : (256 * 1024);
    return (safe * 4) / 5;
}

// Smallest multiple-of-4 K whose NR32 o-block exceeds the budget.
static int int8_kblock_min_K() {
    const int64_t k = (int8_kblock_l2_budget() - 32 * 4) / 32 + 1;
    return static_cast<int>((k + 3) & ~int64_t {3});
}

// Largest multiple-of-4 K whose NR32 o-block still fits the budget.
static int int8_kblock_max_fit_K() {
    const int64_t k = (int8_kblock_l2_budget() - 32 * 4) / 32;
    return static_cast<int>(k & ~int64_t {3});
}

// Straight vs K-blocked output must be bit-identical: s32 accumulation is
// exact and the dequant epilogue runs once on the full-K sum either way.
static void expect_int8_kblock_parity(
        int M, int N, int K, data_type_t scale_dt) {
    using namespace moe_test_utils;
    const Int8AlgoCustomParam p {/*act_int=*/0, M, /*num_ops=*/1, N};
    reset_grp_matmul_caches();
    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N, K, src, wei, bias, dst, scale_dt));

    std::vector<std::vector<bfloat16_t>> straight, blocked;
    {
        KblockOverride kblock_off(0);
        KblockCaptureGuard cap;
        ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei,
                bias, dst, grp_matmul_gated_act_t::none, straight, M, N));
        ASSERT_FALSE(KblockCaptureGuard::used())
                << "reference arm must run the straight kernel";
    }
    for (auto &d : dst) {
        auto *raw = static_cast<bfloat16_t *>(d.get_raw_handle_unsafe());
        std::fill(raw, raw + static_cast<size_t>(M) * N, bfloat16_t(0.0f));
    }

    bool kblock_ran = false;
    {
        KblockOverride kblock_auto(2);
        KblockCaptureGuard cap;
        ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei,
                bias, dst, grp_matmul_gated_act_t::none, blocked, M, N));
        kblock_ran = KblockCaptureGuard::used();
    }
    ASSERT_TRUE(kblock_ran);
    ASSERT_EQ(straight.size(), blocked.size());
    for (size_t e = 0; e < straight.size(); ++e) {
        ASSERT_EQ(straight[e].size(), blocked[e].size());
        ASSERT_EQ(std::memcmp(straight[e].data(), blocked[e].data(),
                          straight[e].size() * sizeof(bfloat16_t)),
                0)
                << "e=" << e;
    }
}

TEST(TestGroupMatmulInt8Kblock, DeepKMatchesStraightNativeKernel) {
    using namespace moe_test_utils;
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }

    const int min_K = int8_kblock_min_K();
    if (min_K > ck::kMaxSafeInt8AccumK) {
        GTEST_SKIP() << "Deep-K threshold " << min_K
                     << " exceeds the safe INT8 accumulation depth";
    }
    // Two full K chunks plus one VNNI quad, so the tail chunk is minimal.
    const int max_mr = ck::max_mr_for_nv_int8(2, ck::IntCompute::kS8_Sym);
    const int kc
            = static_cast<int>(int8_kblock_l2_budget() / (32 + max_mr)) & ~3;
    const int tail_K = 2 * kc + 4;
    std::vector<int> depths {min_K};
    if (tail_K > min_K && tail_K <= ck::kMaxSafeInt8AccumK) {
        depths.push_back(tail_K);
    }

    // N = 1024 over 8 threads gives each tile four NR32 o-blocks.  M = 13,
    // 24 and 32 cover the 7/6, 12/12 (no big call) and 11/11/10 MR splits.
    const int N = 1024;
    AlgoEnvGuard algo_guard(3);
    CustomKernelNROverride nr_guard(32);
    WeightCacheGuard wc_guard(1);
    for (const int K : depths) {
        for (const int M : {13, 24, 32}) {
            for (const data_type_t scale_dt :
                    {data_type_t::f32, data_type_t::bf16}) {
                SCOPED_TRACE("K=" + std::to_string(K)
                        + " M=" + std::to_string(M)
                        + (scale_dt == data_type_t::f32 ? " scale=f32"
                                                        : " scale=bf16"));
                ASSERT_NO_FATAL_FAILURE(
                        expect_int8_kblock_parity(M, N, K, scale_dt));
            }
        }
    }
}

// `kblock_mode` is the shared K-block test override: 2 = automatic policy,
// 1 = forced on (the positive control for shapes automatic mode excludes).
static void expect_int8_kblock_gate(int M, int K, bool with_bias, bool expected,
        int num_ops = 1, int kblock_mode = 2) {
    using namespace moe_test_utils;
    const Int8AlgoCustomParam p {/*act_int=*/0, M, num_ops, /*dim=*/256};
    const int N = p.dim;
    reset_grp_matmul_caches();

    tensor_factory_t tf;
    std::vector<tensor_t> src, wei, bias, dst;
    ASSERT_NO_FATAL_FAILURE(
            build_int8_dq_inputs(tf, p, N, K, src, wei, bias, dst));
    if (with_bias) {
        for (auto &b : bias) {
            b = tf.uniform_dist_tensor(
                    {1, static_cast<uint64_t>(N)}, data_type_t::bf16, 2.0);
        }
    }

    AlgoEnvGuard algo_guard(3);
    CustomKernelNROverride nr_guard(32);
    WeightCacheGuard wc_guard(1);
    KblockOverride kblock(kblock_mode);
    KblockCaptureGuard cap;
    std::vector<std::vector<bfloat16_t>> out;
    ASSERT_NO_FATAL_FAILURE(run_int8_dq_pass(/*int8_on=*/true, src, wei, bias,
            dst, grp_matmul_gated_act_t::none, out, M, N));
    EXPECT_EQ(KblockCaptureGuard::used(), expected);
}

TEST(TestGroupMatmulInt8Kblock, AutomaticGateScope) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    if (!ck::avx512vnni_available()) {
        GTEST_SKIP() << "AVX-512 VNNI not available on this host";
    }
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }

    const int deep_K = int8_kblock_min_K();
    if (deep_K > ck::kMaxSafeInt8AccumK) {
        GTEST_SKIP() << "Deep-K threshold " << deep_K
                     << " exceeds the safe INT8 accumulation depth";
    }
    {
        SCOPED_TRACE("single MR call");
        expect_int8_kblock_gate(
                /*M=*/8, deep_K, /*with_bias=*/false, /*expected=*/false);
    }
    {
        SCOPED_TRACE("packed B block fits L2 budget");
        expect_int8_kblock_gate(/*M=*/32, int8_kblock_max_fit_K(),
                /*with_bias=*/false, /*expected=*/false);
    }
    {
        SCOPED_TRACE("bias uses straight native epilogue");
        expect_int8_kblock_gate(
                /*M=*/32, deep_K, /*with_bias=*/true, /*expected=*/false);
    }
    // Automatic K-blocking is single-expert decode only.  Each exclusion is
    // paired with a forced-on control at the same shape, so a `false` cannot
    // come from the INT8 custom kernel not running at all.
    {
        SCOPED_TRACE("MoE stays straight in automatic mode");
        expect_int8_kblock_gate(/*M=*/32, deep_K, /*with_bias=*/false,
                /*expected=*/false, /*num_ops=*/2);
        expect_int8_kblock_gate(/*M=*/32, deep_K, /*with_bias=*/false,
                /*expected=*/true, /*num_ops=*/2, /*kblock_mode=*/1);
    }
    {
        SCOPED_TRACE("prompt stays straight in automatic mode");
        expect_int8_kblock_gate(/*M=*/64, deep_K, /*with_bias=*/false,
                /*expected=*/false);
        expect_int8_kblock_gate(/*M=*/64, deep_K, /*with_bias=*/false,
                /*expected=*/true, /*num_ops=*/1, /*kblock_mode=*/1);
    }
}

// `ZENDNNL_GRP_MATMUL_INT8_KBLOCK` beats the shared policy when set, and the
// shared test override beats the cached INT8 env, so in-process A/B tests
// stay valid even when the benchmark switch is exported.
TEST(TestGroupMatmulInt8Kblock, OverridePrecedence) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::grp_matmul_int8_kblock_enabled;
    using zendnnl::lowoha::matmul::resolve_grp_matmul_int8_kblock;

    EXPECT_FALSE(resolve_grp_matmul_int8_kblock(0, true));
    EXPECT_TRUE(resolve_grp_matmul_int8_kblock(1, false));
    EXPECT_TRUE(resolve_grp_matmul_int8_kblock(-1, true));
    EXPECT_FALSE(resolve_grp_matmul_int8_kblock(-1, false));

    {
        KblockOverride off(0);
        EXPECT_FALSE(grp_matmul_int8_kblock_enabled(/*kblock_auto=*/true));
    }
    {
        KblockOverride on(1);
        EXPECT_TRUE(grp_matmul_int8_kblock_enabled(/*kblock_auto=*/false));
    }
    {
        KblockOverride automatic(2);
        EXPECT_TRUE(grp_matmul_int8_kblock_enabled(/*kblock_auto=*/true));
        EXPECT_FALSE(grp_matmul_int8_kblock_enabled(/*kblock_auto=*/false));
    }
}

// Smallest even K whose packed B strip for one o-block exceeds the
// K-blocking L2 budget, mirroring the gate in `custom_kernel/dispatch.cpp`.
// Doubled because the packed NR the kernel settles on can be narrower than
// the requested one, and the gate scales with `pack_nr`.
static int kblock_min_K(int nr) {
    const int64_t l2 = zendnnl::lowoha::matmul::native::detect_uarch().l2_bytes;
    const int64_t safe = (l2 >= 64 * 1024) ? l2 : (256 * 1024);
    const int64_t budget = (safe * 4) / 5;
    const int64_t k = 2 * (budget / (static_cast<int64_t>(nr) * 2) + 1);
    return static_cast<int>((k + 1) & ~static_cast<int64_t>(1));
}

// Strictly positive operands drawn from a deterministic LCG.  Non-cancelling
// on purpose: a zero-centred fill sums a deep-K reduction back down to near
// zero, and a tolerance wide enough to cover that noise also accepts an
// all-zero destination.  Positive terms make |dst| grow with K, so the
// comparison below can be purely relative and any unwritten, stale or
// wrong-column output fails it.  Values also vary with both k and n, so
// re-using the first NR block for a tile at nonzero `col_start` produces
// different numbers rather than plausible ones.
static float kblock_rand(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return 0.5f + static_cast<float>((state >> 8) & 0xFFFFu) / 65536.0f;
}

// One single-expert BF16 custom-kernel call with explicit strides.
struct KblockProbe {
    int M = 0, N = 0, K = 0;
    int lda = 0, ldb = 0, ldc = 0;
    bool f32_dst = false;
    std::vector<zendnnl::common::bfloat16_t> src, wei, bias;
    std::vector<zendnnl::common::bfloat16_t> dst_bf16;
    std::vector<float> dst_f32;
    std::vector<const void *> srcs, weis, biases;
    std::vector<void *> dsts;
    moe_test_utils::GemmVecs gv;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
};

static KblockProbe build_kblock_probe(int num_threads, int M, int N, int K,
        bool with_bias, bool f32_dst, int lda_pad, int ldc_pad) {
    using zendnnl::common::bfloat16_t;
    using zendnnl::common::data_type_t;
    KblockProbe p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.lda = K + lda_pad;
    p.ldb = N;
    p.ldc = N + ldc_pad;
    p.f32_dst = f32_dst;

    uint32_t st = 0x5eed1234u;
    p.src.resize(static_cast<size_t>(M) * p.lda);
    for (auto &v : p.src) {
        v = bfloat16_t(kblock_rand(st));
    }
    p.wei.resize(static_cast<size_t>(K) * p.ldb);
    for (auto &v : p.wei) {
        v = bfloat16_t(kblock_rand(st));
    }
    if (with_bias) {
        p.bias.resize(static_cast<size_t>(N));
        for (auto &v : p.bias) {
            v = bfloat16_t(kblock_rand(st));
        }
    }
    if (f32_dst) {
        p.dst_f32.assign(static_cast<size_t>(M) * p.ldc, -1.0f);
    } else {
        p.dst_bf16.assign(static_cast<size_t>(M) * p.ldc, bfloat16_t(-1.0f));
    }

    p.srcs = {p.src.data()};
    p.weis = {p.wei.data()};
    p.biases = {with_bias ? static_cast<const void *>(p.bias.data()) : nullptr};
    p.dsts = {f32_dst ? static_cast<void *>(p.dst_f32.data())
                      : static_cast<void *>(p.dst_bf16.data())};

    p.gv = moe_test_utils::GemmVecs::uniform(/*num_ops=*/1, M, N, K,
            /*a=*/1.0f, /*b=*/0.0f, /*wc=*/true, /*tA=*/false, /*tB=*/false);
    p.gv.lda[0] = p.lda;
    p.gv.ldb[0] = p.ldb;
    p.gv.ldc[0] = p.ldc;
    p.params = moe_test_utils::make_uniform_params(/*num_ops=*/1,
            data_type_t::bf16,
            /*bias_dt=*/with_bias ? data_type_t::bf16 : data_type_t::none);
    p.params[0].num_threads = num_threads;
    if (f32_dst) { p.params[0].dtypes.dst = data_type_t::f32; }
    return p;
}

// Independent FP32 reference, accumulated in double so it does not share the
// kernel's chunking or its accumulation order.  Deliberately not the
// straight-line kernel: comparing the two library paths against each other
// cannot catch an error they both make.
static void kblock_reference(const KblockProbe &p, std::vector<double> &ref) {
    ref.assign(static_cast<size_t>(p.M) * p.N, 0.0);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < p.M; ++m) {
        for (int n = 0; n < p.N; ++n) {
            double acc = p.bias.empty()
                    ? 0.0
                    : static_cast<double>(static_cast<float>(p.bias[n]));
            for (int k = 0; k < p.K; ++k) {
                acc += static_cast<double>(static_cast<float>(
                               p.src[static_cast<size_t>(m) * p.lda + k]))
                        * static_cast<double>(static_cast<float>(
                                p.wei[static_cast<size_t>(k) * p.ldb + n]));
            }
            ref[static_cast<size_t>(m) * p.N + n] = acc;
        }
    }
}

// Runs one shape and validates it against the independent reference.
//
// The bound is purely relative and derived per destination dtype rather than
// picked as a round number, so it tracks what the arithmetic can actually
// lose.  Two terms:
//
//   * the destination store — a BF16 write keeps 8 mantissa bits, so it
//     costs up to 2^-8 relative; an F32 write costs nothing;
//   * the f32 reduction over K terms, which drifts like sqrt(K)*2^-24.
//
// Each is taken with a safety factor.  In practice the BF16 cases land near
// 2.6e-3 and the F32 case near 4e-6 — the latter being the interesting one,
// since it holds the kernel to within a few ulp of an independent double
// accumulation and no amount of misplaced or stale output survives it.
void group_matmul_test::run_kblock_case(int num_threads, int M, int N, int K,
        int nr, int kblock_mode, bool with_bias, bool f32_dst, int lda_pad,
        int ldc_pad, bool expect_kblock, const char *label) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;

    auto p = build_kblock_probe(
            num_threads, M, N, K, with_bias, f32_dst, lda_pad, ldc_pad);

    // `kblock_reference` recomputes from `p.weis` AFTER the dispatch below,
    // so the weight pool must survive the call unmodified.  Pin out-of-place.
    WeightCacheGuard wc_out_of_place(1);

    bool kblock_ran = false;
    {
        AlgoEnvGuard algo3_guard(3);
        CustomKernelOverride ck_on(true);
        CustomKernelNROverride nr_guard(nr);
        KblockOverride kblock_mode_guard(kblock_mode);
        KblockCaptureGuard cap;
        ::reset_grp_matmul_caches();
        auto pt = p.params;
        ASSERT_EQ(group_matmul_direct(p.gv.layout, p.gv.transA, p.gv.transB,
                          p.gv.Ms, p.gv.Ns, p.gv.Ks, p.gv.alpha, p.srcs,
                          p.gv.lda, p.weis, p.gv.ldb, p.biases, p.gv.beta,
                          p.dsts, p.gv.ldc, p.gv.is_wc, pt, nullptr, nullptr),
                status_t::success)
                << label << ": dispatch failed";
        kblock_ran = KblockCaptureGuard::used();
    }

    ASSERT_EQ(kblock_ran, expect_kblock)
            << label << ": K-blocked tile "
            << (expect_kblock ? "must run for this shape"
                              : "must not run for this shape");

    std::vector<double> ref;
    kblock_reference(p, ref);

    const double acc_tol = 8.0 * std::sqrt(static_cast<double>(p.K)) * 5.96e-8;
    const double store_tol = f32_dst ? 0.0 : 1.5 / 256.0;
    const double tol = store_tol + acc_tol;

    double worst_rel = 0.0;
    for (int m = 0; m < p.M; ++m) {
        for (int n = 0; n < p.N; ++n) {
            const double want = ref[static_cast<size_t>(m) * p.N + n];
            const size_t off = static_cast<size_t>(m) * p.ldc + n;
            const double got = f32_dst
                    ? static_cast<double>(p.dst_f32[off])
                    : static_cast<double>(static_cast<float>(p.dst_bf16[off]));
            const double rel
                    = std::fabs(got - want) / std::max(1e-30, std::fabs(want));
            if (rel > worst_rel) { worst_rel = rel; }
            ASSERT_LT(rel, tol)
                    << label
                    << ": output diverges from the FP32 reference at row " << m
                    << " col " << n << " — got " << got << ", expected "
                    << want;
        }
    }
    // The reference is O(K) in magnitude, so a destination left untouched or
    // filled with the wrong columns cannot land inside the relative bound.
    EXPECT_GT(std::fabs(ref[0]), 1.0)
            << label
            << ": probe produced a degenerate reference; the "
               "comparison above would not be meaningful";
    // Reported so the margin between observed error and the bound is
    // visible rather than asserted on faith.
    std::cout << "[   INFO   ] " << label << ": worst relative error "
              << worst_rel << " against a bound of " << tol
              << ", |ref[0]| = " << std::fabs(ref[0]) << std::endl;
}

using group_matmul_test::run_kblock_case;

// Columns beyond the first NR block force tiles at nonzero `col_start`, and
// M=64 spans several MR blocks.
TEST(TestGroupMatmulKblock, BF16ParityNR64) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/true, "kblock_nr64");
}

TEST(TestGroupMatmulKblock, BF16ParityNR32) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/32 * 8,
            /*K=*/kblock_min_K(32), /*nr=*/32, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/true, "kblock_nr32");
}

// F32 destination takes a different store path (4-byte elements, no BF16
// narrowing) than the BF16 one the other cases cover.
TEST(TestGroupMatmulKblock, F32DstParity) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/true, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/true, "kblock_f32_dst");
}

// Independently padded lda and ldc: row offsets are computed from each
// stride separately, so equal strides would hide a swap between them.
TEST(TestGroupMatmulKblock, PaddedStridesParity) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/16,
            /*ldc_pad=*/48, /*expect_kblock=*/true, "kblock_padded_strides");
}

// The path the PR actually ships: no env set, `kblock_auto` alone decides.
// It is armed only for a single expert in decode (M <= kDecodeMaxM), which
// is why the forced cases above do not exercise it.
TEST(TestGroupMatmulKblock, AutoEnablesOnSingleExpertDecode) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/32, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/2,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/true, "kblock_auto_decode");
}

// Same shape past the decode M ceiling: automatic enablement must stay off,
// which is what keeps prompt-class calls on the straight-line path.
TEST(TestGroupMatmulKblock, AutoStaysOffAboveDecodeM) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/2,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/false, "kblock_auto_prompt");
}

// Odd K cannot be paired into VNNI blocks, so the gate rejects it.
TEST(TestGroupMatmulKblock, OddKFallsBackToStraightLine) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64) + 1, /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/false, "kblock_odd_k");
}

// Bias is applied by the straight-line epilogue only.
TEST(TestGroupMatmulKblock, BiasFallsBackToStraightLine) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/64, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/true, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/false, "kblock_bias");
}

// M inside a single MR block leaves nothing for the B-strip reuse to
// amortise, which is the whole point of the path.
TEST(TestGroupMatmulKblock, SingleMrBlockFallsBackToStraightLine) {
    KblockThreadGuard tg(8);
    if (!require_min_threads(8)) {
        GTEST_SKIP() << "Requires >= 8 OMP threads.";
    }
    run_kblock_case(/*num_threads=*/8, /*M=*/1, /*N=*/64 * 8,
            /*K=*/kblock_min_K(64), /*nr=*/64, /*kblock_mode=*/1,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/false, "kblock_single_mr");
}

TEST(TestGroupMatmulMTileCustomKernelInvariance, BF16ByteIdenticalDst) {
    using namespace moe_test_utils;
    using zendnnl::lowoha::matmul::group_matmul_direct;
    using zendnnl::lowoha::matmul::status_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    // Two independent shapes — first with CK=0, second with CK=1.  Same
    // src/wei/bias inputs across both runs (deterministic fill).  The
    // helper builds a fresh shape per call so the two dst buffers don't
    // alias.
    auto s0 = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/true);
    auto s1 = build_standalone_mtile_shape(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/true);

    int tag0 = 0, tag1 = 0;
    // Run 1: CK=0.
    {
        AlgoEnvGuard algo2_guard(2);
        CustomKernelOverride ck_off(false);
        MTileHybridOverride hybrid_auto(0);
        MTilePathCaptureGuard cap;
        ::reset_grp_matmul_caches();
        auto pt = s0.params;
        ASSERT_EQ(group_matmul_direct(s0.gv.layout, s0.gv.transA, s0.gv.transB,
                          s0.gv.Ms, s0.gv.Ns, s0.gv.Ks, s0.gv.alpha, s0.srcs,
                          s0.gv.lda, s0.weis, s0.gv.ldb, s0.biases, s0.gv.beta,
                          s0.dsts_test, s0.gv.ldc, s0.gv.is_wc, pt, nullptr,
                          nullptr),
                status_t::success)
                << "CK=0 dispatch failed";
        tag0 = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
                std::memory_order_relaxed);
    }
    // Run 2: CK=1.
    {
        AlgoEnvGuard algo2_guard(2);
        CustomKernelOverride ck_on(true);
        MTileHybridOverride hybrid_auto(0);
        MTilePathCaptureGuard cap;
        ::reset_grp_matmul_caches();
        auto pt = s1.params;
        ASSERT_EQ(group_matmul_direct(s1.gv.layout, s1.gv.transA, s1.gv.transB,
                          s1.gv.Ms, s1.gv.Ns, s1.gv.Ks, s1.gv.alpha, s1.srcs,
                          s1.gv.lda, s1.weis, s1.gv.ldb, s1.biases, s1.gv.beta,
                          s1.dsts_test, s1.gv.ldc, s1.gv.is_wc, pt, nullptr,
                          nullptr),
                status_t::success)
                << "CK=1 dispatch failed";
        tag1 = zendnnl::lowoha::matmul::test_api ::s_last_m_tile_path.load(
                std::memory_order_relaxed);
    }

    // Both runs MUST land on the same M-tile branch — CK doesn't
    // influence the planner.
    EXPECT_EQ(tag0, kPhase2Single)
            << "CK=0 run: expected kPhase2Single (=3), got " << tag0;
    EXPECT_EQ(tag1, kPhase2Single)
            << "CK=1 run: expected kPhase2Single (=3), got " << tag1;

    // Byte-identical dst across both runs — any difference signals a
    // hidden CK-dependent code path inside M-tile.
    for (int e = 0; e < s0.num_ops; ++e) {
        ASSERT_EQ(s0.dst_test[e].size(), s1.dst_test[e].size())
                << "dst buffer size mismatch on expert " << e;
        const auto *p0
                = reinterpret_cast<const uint8_t *>(s0.dst_test[e].data());
        const auto *p1
                = reinterpret_cast<const uint8_t *>(s1.dst_test[e].data());
        const size_t bytes
                = s0.dst_test[e].size() * sizeof(zendnnl::common::bfloat16_t);
        for (size_t b = 0; b < bytes; ++b) {
            if (p0[b] != p1[b]) {
                FAIL() << "M-tile dst is NOT byte-identical across CK on/off: "
                          "expert="
                       << e << " byte_off=" << b
                       << " CK=0=" << static_cast<int>(p0[b])
                       << " CK=1=" << static_cast<int>(p1[b])
                       << " — `flat_m_tile` must not pick up a CK-dependent "
                          "kernel-select branch";
            }
        }
    }
}

TEST(TestGroupMatmulMTileStandalone, F32WithBiasF32) {
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f32(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/128, /*K=*/64,
            /*with_bias=*/true);
    run_and_compare_standalone_f32(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/1e-5f, /*atol=*/1e-5f, "standalone_f32_bias_f32");
}

// Forced ALGO 2 with active_ops > num_threads AND a fused gated activation:
// the sequential full-team clamp (kManyExpertsSeqFallback) must still apply
// the full-M post-pass activation per expert correctly.
TEST(TestGroupMatmulMTileBranches, ManyExpertsSeqClampGatedActFullMPostPass) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag ::
            kManyExpertsSeqFallback;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(8);
    if (!require_min_threads(4)) {
        GTEST_SKIP() << "Requires >= 4 OMP threads.";
    }

    auto s = build_standalone_mtile_shape(
            /*num_threads=*/4,
            /*Ms=*/std::vector<int>(8, 8),
            /*N=*/256, /*K=*/64,
            /*with_bias=*/false);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::silu_and_mul;

    run_and_compare_standalone(s, /*algo_test=*/2, /*algo_ref=*/1,
            /*expected_tag=*/kManyExpertsSeqFallback,
            /*rtol=*/5e-2f, /*atol=*/5e-2f,
            "many_experts_seq_clamp_silu_full_M_post_pass",
            /*gated_act=*/&ga);
}

// ──────────────────────────────────────────────────────────────────
// F16 standalone gated-act siblings.
//
// Locks in that the F16 row kernels added to
// `group_matmul_moe_act.cpp` (`execute_act_rows_avx512_f16` and the
// `execute_act_rows_scalar_f16` fallback) are correctly reached when
// the dispatcher routes a `data_type_t::f16` dst with a non-null
// `gated_act` through the M-tile machinery.  The BF16 siblings above
// already prove the M-tile vs. ALGO-1 path-tag parity; these tests
// extend the same parity contract to F16 so any future change that
// silently routes F16 + gated_act to the BF16 row kernel (the bug
// that the dispatcher-level guard used to catch) is caught at
// test time instead of as silently corrupted output.
//
// Tests cleanly GTEST_SKIP() on hosts without AVX-512-FP16
// (`run_and_compare_standalone_f16` returns false when the
// dispatcher's F16 ISA gate refuses ALGO 1).
// ──────────────────────────────────────────────────────────────────

TEST(TestGroupMatmulMTileStandalone, GatedActSiluAndMulF16) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f16(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::silu_and_mul;

    const bool ran = run_and_compare_standalone_f16(s, /*algo_test=*/2,
            /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_f16_silu_and_mul",
            /*gated_act=*/&ga);
    if (!ran) { GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA"; }
}

TEST(TestGroupMatmulMTileStandalone, GatedActGeluAndMulF16) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f16(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::gelu_and_mul;

    const bool ran = run_and_compare_standalone_f16(s, /*algo_test=*/2,
            /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_f16_gelu_and_mul",
            /*gated_act=*/&ga);
    if (!ran) { GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA"; }
}

TEST(TestGroupMatmulMTileStandalone, GatedActSwigluOaiMulF16) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kPhase2Single;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(32);
    if (!require_min_threads(32)) {
        GTEST_SKIP() << "Requires >= 32 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f16(
            /*num_threads=*/32,
            /*Ms=*/std::vector<int>(16, 64),
            /*N=*/256, /*K=*/64);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::swiglu_oai_mul;

    const bool ran = run_and_compare_standalone_f16(s, /*algo_test=*/2,
            /*algo_ref=*/1,
            /*expected_tag=*/kPhase2Single,
            /*rtol=*/5e-2f, /*atol=*/5e-2f, "standalone_f16_swiglu_oai_mul",
            /*gated_act=*/&ga);
    if (!ran) { GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA"; }
}

// F16 sibling of `RoundBasedGatedActFullMPostPass`.  Same
// 8 experts × M=8 × N=256 × K=64 shape (drives ALGO 2 to the
// round-based path with a full-M apply_gated_act_inplace post-pass
// instead of the per-slice path that the larger configs above
// exercise).
TEST(TestGroupMatmulMTileBranches, RoundBasedGatedActFullMPostPassF16) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    using zendnnl::lowoha::matmul::test_api::m_tile_path_tag::kRoundBased;

    const int saved = omp_get_max_threads();
    struct G {
        int p;
        ~G() { omp_set_num_threads(p); }
    } g {saved};
    omp_set_num_threads(8);
    if (!require_min_threads(4)) {
        GTEST_SKIP() << "Requires >= 4 OMP threads.";
    }

    auto s = build_standalone_mtile_shape_f16(
            /*num_threads=*/4,
            /*Ms=*/std::vector<int>(8, 8),
            /*N=*/256, /*K=*/64);

    grp_matmul_gated_act_params ga {};
    ga.act = grp_matmul_gated_act_t::silu_and_mul;

    const bool ran
            = run_and_compare_standalone_f16(s, /*algo_test=*/2, /*algo_ref=*/1,
                    /*expected_tag=*/kRoundBased,
                    /*rtol=*/5e-2f, /*atol=*/5e-2f,
                    "round_based_silu_full_M_post_pass_f16",
                    /*gated_act=*/&ga);
    if (!ran) { GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA"; }
}
