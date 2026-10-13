/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *******************************************************************************/

#include <string>
#include <utility>

#include <omp.h>

#include "../../kblock_test_hooks.hpp"
#include "../ck_test_helpers.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/qwen/bf16_decode_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/qwen/bf16_decode_policy.hpp"

namespace {

namespace ck = ck_test::ck;
using ck_test::grp_matmul_gated_act_t;
using ck_test::status_t;

inline ck_test::PrepCallCase qwen_case(std::string label) {
    ck_test::PrepCallCase c {};
    c.label = std::move(label);
    return c;
}

struct QwenThreadGuard {
    int previous;
    explicit QwenThreadGuard(int count) : previous(omp_get_max_threads()) {
        omp_set_num_threads(count);
    }
    ~QwenThreadGuard() { omp_set_num_threads(previous); }
};

bool qwen_has_threads(int needed) {
    int actual = 0;
#pragma omp parallel
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    return actual >= needed;
}

status_t run_prepare_with_qwen_policy(const ck_test::PrepCallCase &c,
        ck_test::PrepCallStorage &storage, ck::CallContext &kctx,
        ck::QwenBf16DecodeRoute *route = nullptr) {
    const status_t status = ck_test::run_prepare(c, storage, kctx);
    ck::QwenBf16DecodeRoute selected = ck::QwenBf16DecodeRoute::none;
    if (status == status_t::success) {
        selected = ck::apply_qwen_bf16_decode_policy(
                kctx, c.act, {c.M}, {c.N}, {c.K});
    }
    if (route != nullptr) *route = selected;
    return status;
}

TEST(QwenBf16DecodePolicy, UsesOnePanelKernelsAboveExistingNTileGate) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen35_decode_plain");
    c.M = 32;
    c.N = 4096;
    c.K = 4096;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, 12);
    EXPECT_TRUE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, UsesOnePanelGatedKernel) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen35_decode_gated");
    c.M = 32;
    c.N = 24576;
    c.K = 4096;
    c.act = grp_matmul_gated_act_t::silu_and_mul;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, 12);
    EXPECT_TRUE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, KeepsPromptOnEstablishedKernel) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen35_prompt_plain");
    c.M = 128;
    c.N = 4096;
    c.K = 4096;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, ck::max_mr_for_nv(kctx.NV));
    EXPECT_FALSE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, KeepsUnknownShapeOnEstablishedKernel) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen_decode_unknown");
    c.M = 32;
    c.N = 2080;
    c.K = 4096;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, ck::max_mr_for_nv(kctx.NV));
    EXPECT_TRUE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, SelectsIsolatedDownProjectionScheduler) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen35_decode_down_projection");
    c.M = 32;
    c.N = 4096;
    c.K = 12288;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ck::QwenBf16DecodeRoute route;
    ASSERT_EQ(run_prepare_with_qwen_policy(c, storage, kctx, &route),
            status_t::success);
    EXPECT_EQ(route, ck::QwenBf16DecodeRoute::qwen35_w2_resume);
    EXPECT_EQ(kctx.max_mr, ck::max_mr_for_nv(kctx.NV));
    EXPECT_TRUE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, KeepsSmallDownProjectionOnEstablishedKernel) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen35_decode_down_projection_small");
    c.M = 8;
    c.N = 4096;
    c.K = 12288;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ck::QwenBf16DecodeRoute route;
    ASSERT_EQ(run_prepare_with_qwen_policy(c, storage, kctx, &route),
            status_t::success);
    EXPECT_EQ(route, ck::QwenBf16DecodeRoute::none);
    EXPECT_TRUE(kctx.kblock_auto);
}

TEST(QwenBf16DecodePolicy, UsesFixedKKernelForQwen36BS8) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen36_decode_input");
    c.M = 8;
    c.N = 12288;
    c.K = 2048;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, 12);
    EXPECT_TRUE(kctx.kblock_auto);
    EXPECT_EQ(kctx.kfn_table[8],
            ck::decode_specialized::select_plain_bf16_fixed_k(8, 2048));
}

TEST(QwenBf16DecodePolicy, UsesFixedKKernelForQwen36Router) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("qwen36_decode_router");
    c.M = 8;
    c.N = 256;
    c.K = 2048;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_TRUE(kctx.kblock_auto);
    EXPECT_EQ(kctx.kfn_table[8],
            ck::decode_specialized::select_plain_bf16_fixed_k(8, 2048));
}

TEST(QwenBf16DecodePolicy, KeepsGptOssOnEstablishedKernel) {
    CK_SKIP_IF_NO_BF16_ISA();
    ::reset_grp_matmul_caches();
    auto c = qwen_case("gpt_oss_decode");
    c.M = 8;
    c.N = 5760;
    c.K = 2880;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(
            run_prepare_with_qwen_policy(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.max_mr, ck::max_mr_for_nv(kctx.NV));
    EXPECT_TRUE(kctx.kblock_auto);
    EXPECT_EQ(kctx.kfn_table[8],
            ck::select_ukernel(
                    8, kctx.NV, ck::ActKind::none, ck::DstDt::kBf16));
}

TEST(QwenBf16DecodeE2E, DownProjectionUsesSidecar) {
    QwenThreadGuard threads(8);
    if (!qwen_has_threads(8)) { GTEST_SKIP() << "Requires >= 8 OMP threads."; }
    group_matmul_test::run_kblock_case(/*num_threads=*/8, /*M=*/32,
            /*N=*/4096, /*K=*/12288, /*nr=*/32, /*kblock_mode=*/2,
            /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
            /*ldc_pad=*/0, /*expect_kblock=*/false, "qwen35_w2_sidecar_m32");
}

TEST(QwenBf16DecodeE2E, DownProjectionHandlesEveryMrTail) {
    QwenThreadGuard threads(8);
    if (!qwen_has_threads(8)) { GTEST_SKIP() << "Requires >= 8 OMP threads."; }
    for (const int M : {31, 9}) {
        group_matmul_test::run_kblock_case(/*num_threads=*/8, M,
                /*N=*/4096, /*K=*/12288, /*nr=*/32, /*kblock_mode=*/2,
                /*with_bias=*/false, /*f32_dst=*/false, /*lda_pad=*/0,
                /*ldc_pad=*/0, /*expect_kblock=*/false,
                M == 31 ? "qwen35_w2_sidecar_tail31"
                        : "qwen35_w2_sidecar_tail9");
    }
}

} // namespace
