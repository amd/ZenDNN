/********************************************************************************
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

/// @file test_group_reorder_e2e.cpp
/// @brief End-to-end model-style flow for the custom-kernel weight prepack:
///        an 8-expert MoE layer prepacks its weights once via
///        `reorder::group_reorder` (moe_custom_kernel mode) into
///        caller-owned VNNI buffers, then runs inference via the
///        `group_matmul` API with `mem_format_b='r'`, which consumes those
///        already-reordered buffers DIRECTLY — no re-pack, no cache.
///
/// Three things are asserted:
///   1. NO RE-PACK: because every weight is `mem_format_b='r'`, the eager
///      ALGO-3 warm (`prepack_for_algo_3`) is skipped, so no prepack
///      invocation is recorded (`stats.valid == false`).
///   2. CK ENGAGED: `group_matmul_direct` returns `status_t::success`.
///      For a prepacked weight that is sufficient proof the custom kernel
///      ran on ALGO 3 — the CK-only-or-fail guard fails any prepacked call
///      that would fall back to a non-CK executor (which would mis-read the
///      VNNI bytes as a raw weight).
///   3. CORRECTNESS: the group_matmul output matches the per-expert
///      reference GEMM (on the ORIGINAL un-packed weights) within BF16
///      tolerance — proving the directly-consumed VNNI buffers are correct.
///
/// The env guards below request ALGO 3 + CK (via CustomKernelOverride,
/// which bypasses the process-cached ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL
/// snapshot a sibling test may have taken). The host-support check at entry
/// skips the test where AVX-512 BF16 is unavailable.
///
/// Two variants live here: the bf16 flow (VDPBF16PS VNNI pack) and an
/// FP16 sibling (native AVX-512-FP16 plain slab, gated on
/// `avx512f16_available()`) that exercises the f16 external-prepack
/// surface (`prepack_weight_into_f16` via group_reorder).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "common/op_config.hpp"
#include "group_matmul_test_helpers.hpp"
#include "gtest_utils.hpp"
#include "moe_test_utils.hpp"

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace {

namespace rdr = zendnnl::lowoha::reorder;
namespace ck = zendnnl::lowoha::matmul::custom_kernel;
namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
namespace mt = moe_test_utils;

// RAII for the library-wide weight-cache toggle so the warm actually
// populates the LRU (custom_kernel prepack no-ops when it is off).
class WeightCacheGuard {
public:
    explicit WeightCacheGuard(int32_t v)
        : prev_(zendnnl::common::matmul_config_t::instance()
                          .get_weight_cache()) {
        zendnnl::common::matmul_config_t::instance().set_weight_cache(v);
    }
    ~WeightCacheGuard() {
        zendnnl::common::matmul_config_t::instance().set_weight_cache(prev_);
    }
    WeightCacheGuard(const WeightCacheGuard &) = delete;
    WeightCacheGuard &operator=(const WeightCacheGuard &) = delete;

private:
    int32_t prev_;
};

// ──────────────────────────────────────────────────────────────────
// 8-expert MoE: warm via group_reorder, infer via group_matmul.
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorderModelE2E, WarmUpThenInferenceFetchesReorderedWeights) {
    if (!ck::dispatch_supported()) {
        GTEST_SKIP() << "AVX-512 BF16 not available; the custom-kernel pack "
                        "path cannot run on this host";
    }
    prepack::clear_fingerprint_cache_for_test();
    reset_grp_matmul_caches();
    prepack::test_api::clear_last_invocation_stats();

    // Force the decode ALGO-3 + custom-kernel path for inference.
    // CustomKernelOverride bypasses the process-cached
    // ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL toggle (a plain EnvVarGuard is read
    // too late), so CK deterministically engages. All restored on exit.
    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride ck_on(true);
    // The warm only populates the LRU when the weight cache is enabled.
    WeightCacheGuard wc_on(1);
    // Arm capture of the dispatch-time prepack probe stats so we can assert
    // the group_matmul ALGO-3 warm HITs the cache group_reorder populated.
    mt::LastInvocationCaptureGuard stats_capture;

    // gpt-oss-style decode shape: small M, K/N multiples of the pack width.
    constexpr int E = 8; // experts
    constexpr int M = 16; // tokens routed to each expert
    constexpr int K = 256; // in features
    constexpr int N = 256; // out features (multiple of pack_nr=32)
    constexpr matmul_algo_t kAlgo = matmul_algo_t::aocl_dlp_blocked;

    tensor_factory_t tf {};
    std::vector<tensor_t> inp(E), wt(E), bias(E), out(E), out_ref(E);
    std::vector<const void *> wptr(E);
    for (int e = 0; e < E; ++e) {
        inp[e] = tf.uniform_dist_tensor({M, K}, data_type_t::bf16, 2.0, false);
        wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::bf16, 2.0, false);
        bias[e] = tensor_t {}; // no bias (CK bf16 none-act path)
        out[e] = tf.uniform_dist_tensor({M, N}, data_type_t::bf16, 2.0);
        out_ref[e] = tf.uniform_dist_tensor({M, N}, data_type_t::bf16, 2.0);
        wptr[e] = wt[e].get_raw_handle_unsafe();
    }

    // ── PREPACK: group_reorder changes each weight's MEMORY FORMAT into a
    //    caller-owned VNNI buffer (no caching).  Two-step contract:
    //    weight_prepack_size() -> allocate -> group_reorder() writes the
    //    VNNI bytes into the caller's dst. ───────────────────────────────
    std::vector<rdr::reorder_params_t> rp(E);
    std::vector<const void *> src_w(E);
    std::vector<void *> prepacked(E, nullptr);
    for (int e = 0; e < E; ++e) {
        rp[e].is_prepack = true;
        rp[e].prepack.algo = matmul_algo_t::moe_custom_kernel;
        rp[e].prepack.wei_dtype = data_type_t::bf16;
        rp[e].prepack.src_dtype = data_type_t::bf16;
        rp[e].prepack.K = K;
        rp[e].prepack.N = N;
        rp[e].prepack.ldb = N; // row-major [K, N]
        rp[e].prepack.transposed = false;
        rp[e].prepack.pack_nr = 0; // auto (plan_pack_nr)

        const size_t bytes = rdr::weight_prepack_size(rp[e]);
        ASSERT_GT(bytes, 0u)
                << "weight_prepack_size returned 0 for expert " << e;
        // 64-byte aligned: the custom-kernel microkernel reads the packed
        // weight with aligned AVX-512 loads (_mm512_load_si512).
        prepacked[e] = zendnnl_aligned_alloc(64, bytes);
        ASSERT_NE(prepacked[e], nullptr);
        src_w[e] = wptr[e]; // original raw weight (read by reorder)
    }
    ASSERT_EQ(rdr::group_reorder(src_w, prepacked, rp),
            zendnnl::memory::status_t::success)
            << "group_reorder (memory-format change) failed";

    // ── INFERENCE: group_matmul over the PREPACKED weights, telling it the
    //    weights are already reordered (mem_format_b='r') so it consumes
    //    them directly and does NOT re-pack. ─────────────────────────────
    std::vector<char> layouts(E, 'r');
    std::vector<bool> transAs(E, false), transBs(E, false);
    std::vector<int> Ms(E, M), Ns(E, N), Ks(E, K);
    std::vector<float> alphas(E, 1.0f), betas(E, 0.0f);
    std::vector<int> ldas(E, K), ldbs(E, N), ldcs(E, N);
    // is_weights_const=true is REQUIRED for the CK path to engage at
    // runtime (prepare_for_call refuses a non-const active expert).
    std::vector<bool> is_wc(E, true);

    std::vector<const void *> srcs(E), weis(E), biases(E, nullptr);
    std::vector<void *> dsts(E);
    std::vector<matmul_params> params(E);
    for (int e = 0; e < E; ++e) {
        srcs[e] = inp[e].get_raw_handle_unsafe();
        weis[e] = prepacked[e]; // the VNNI-packed weight from group_reorder
        dsts[e] = out[e].get_raw_handle_unsafe();
        params[e].dtypes.src = data_type_t::bf16;
        params[e].dtypes.wei = data_type_t::bf16;
        params[e].dtypes.dst = data_type_t::bf16;
        params[e].dtypes.bias = data_type_t::none;
        params[e].mem_format_b = 'r'; // already reordered -> consume directly
        // 'r' is ambiguous on its own; lowoha_algo disambiguates the packed
        // layout: moe_custom_kernel => the CK VNNI buffer group_reorder
        // produced above (vs aocl_dlp_blocked => AOCL DLP blocked layout).
        params[e].lowoha_algo = matmul_algo_t::moe_custom_kernel;
        params[e].num_threads = 0;
    }

    status_t st = group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks,
            alphas, srcs, ldas, weis, ldbs, biases, betas, dsts, ldcs, is_wc,
            params, /*moe_postop=*/nullptr);
    ASSERT_EQ(st, status_t::success)
            << "group_matmul_direct (mem_format_b='r') failed";

    // Contract check: a pre-reordered (mem_format_b='r') weight is consumed
    // DIRECTLY by the custom kernel and must NOT be re-packed.  The eager
    // ALGO-3 warm (`prepack_for_algo_3`, the only producer of these stats)
    // is therefore skipped — so NO prepack invocation should be recorded.
    // That CK actually engaged on ALGO 3 is already guaranteed by the
    // `status_t::success` assertion above: the CK-only-or-fail guard fails
    // any prepacked call that would fall back to a non-CK executor.
    auto stats = prepack::test_api::get_last_invocation_stats();
    EXPECT_FALSE(stats.valid)
            << "eager ALGO-3 warm ran for a prepacked (mem_format_b='r') "
               "weight; "
               "it must be skipped — prepacked weights are consumed directly, "
               "never re-packed";

    // Reference: per-expert GEMM on the ORIGINAL (un-packed) weights.
    status_t ref_st = status_t::success;
    for (int e = 0; e < E && ref_st == status_t::success; ++e) {
        std::vector<post_op_type_t> ref_po;
        std::vector<tensor_t> bin;
        ref_st = matmul_kernel_test(inp[e], wt[e], bias[e], out_ref[e], ref_po,
                bin,
                /*use_LOWOHA=*/true, kAlgo,
                /*alpha=*/1.0f, /*beta=*/0.0f, /*use_reference=*/true);
    }
    ASSERT_EQ(ref_st, status_t::success) << "reference GEMM failed";

    bool ok = true;
    for (int e = 0; e < E && ok; ++e) {
        compare_tensor_2D_matrix(out[e], out_ref[e], M, N, K, rtol_bf16,
                epsilon_bf16, ok,
                /*enable_f32_relaxation=*/false, /*alpha=*/1.0f);
    }
    EXPECT_TRUE(ok)
            << "group_matmul output mismatch vs reference (prepacked weights)";

    for (int e = 0; e < E; ++e)
        zendnnl_aligned_free(prepacked[e]);
}

// ──────────────────────────────────────────────────────────────────
// FP16 sibling of the bf16 e2e above.  Same prepack-at-load → infer
// flow, but the weight family is native AVX-512-FP16 (plain
// [O/pack_nr][K][pack_nr] slab, no VNNI K-pair doubling).  Exercises
// the f16 external-prepack surface (`prepack_weight_into_f16` reached
// via group_reorder(moe_custom_kernel, wei_dtype=f16)) and the
// dispatcher's f16 caller-prepacked aliasing.
//
// Gated on `avx512f16_available()` (CPUID + toolchain FP16 intrinsics),
// NOT `dispatch_supported()` (which only checks bf16): a bf16-only host
// has no FP16 microkernel, so the CK f16 path cannot engage and the
// CK-only-or-fail guard would (correctly) fail the prepacked call.
// ──────────────────────────────────────────────────────────────────
TEST(GroupReorderModelE2E, F16WarmUpThenInferenceFetchesReorderedWeights) {
    if (!ck::avx512f16_available()) {
        GTEST_SKIP() << "AVX-512-FP16 not available (CPU or toolchain); the "
                        "custom-kernel FP16 pack path cannot run on this host";
    }
    prepack::clear_fingerprint_cache_for_test();
    reset_grp_matmul_caches();
    prepack::test_api::clear_last_invocation_stats();

    mt::AlgoEnvGuard algo3(3);
    mt::CustomKernelOverride ck_on(true);
    WeightCacheGuard wc_on(1);
    mt::LastInvocationCaptureGuard stats_capture;

    constexpr int E = 8; // experts
    constexpr int M = 16; // tokens routed to each expert
    constexpr int K = 256; // in features
    constexpr int N = 256; // out features (multiple of pack_nr=32)
    constexpr matmul_algo_t kAlgo = matmul_algo_t::aocl_dlp_blocked;

    // FP16 accumulates natively (`_mm512_fmadd_ph`), so the tolerance
    // band is wider than bf16 — mirror the f16 group basic test
    // (`test_basic.cpp`: 5x rtol_bf16 / 16x epsilon_bf16).
    const float f16_rtol = 5.0f * rtol_bf16;
    const float f16_eps = 16.0f * epsilon_bf16;

    tensor_factory_t tf {};
    std::vector<tensor_t> inp(E), wt(E), bias(E), out(E), out_ref(E);
    std::vector<const void *> wptr(E);
    for (int e = 0; e < E; ++e) {
        inp[e] = tf.uniform_dist_tensor({M, K}, data_type_t::f16, 2.0, false);
        wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::f16, 2.0, false);
        bias[e] = tensor_t {}; // no bias (CK f16 none-act path)
        out[e] = tf.uniform_dist_tensor({M, N}, data_type_t::f16, 2.0);
        out_ref[e] = tf.uniform_dist_tensor({M, N}, data_type_t::f16, 2.0);
        wptr[e] = wt[e].get_raw_handle_unsafe();
    }

    // ── PREPACK: group_reorder → caller-owned FP16 VNNI slab. ──────────
    std::vector<rdr::reorder_params_t> rp(E);
    std::vector<const void *> src_w(E);
    std::vector<void *> prepacked(E, nullptr);
    for (int e = 0; e < E; ++e) {
        rp[e].is_prepack = true;
        rp[e].prepack.algo = matmul_algo_t::moe_custom_kernel;
        rp[e].prepack.wei_dtype = data_type_t::f16;
        rp[e].prepack.src_dtype = data_type_t::f16;
        rp[e].prepack.K = K;
        rp[e].prepack.N = N;
        rp[e].prepack.ldb = N; // row-major [K, N]
        rp[e].prepack.transposed = false;
        rp[e].prepack.pack_nr = 0; // auto (plan_pack_nr)

        const size_t bytes = rdr::weight_prepack_size(rp[e]);
        ASSERT_GT(bytes, 0u)
                << "weight_prepack_size (f16) returned 0 for expert " << e;
        prepacked[e] = zendnnl_aligned_alloc(64, bytes);
        ASSERT_NE(prepacked[e], nullptr);
        src_w[e] = wptr[e];
    }
    ASSERT_EQ(rdr::group_reorder(src_w, prepacked, rp),
            zendnnl::memory::status_t::success)
            << "group_reorder (f16 memory-format change) failed";

    // ── INFERENCE over the prepacked f16 weights (mem_format_b='r'). ────
    std::vector<char> layouts(E, 'r');
    std::vector<bool> transAs(E, false), transBs(E, false);
    std::vector<int> Ms(E, M), Ns(E, N), Ks(E, K);
    std::vector<float> alphas(E, 1.0f), betas(E, 0.0f);
    std::vector<int> ldas(E, K), ldbs(E, N), ldcs(E, N);
    std::vector<bool> is_wc(E, true);

    std::vector<const void *> srcs(E), weis(E), biases(E, nullptr);
    std::vector<void *> dsts(E);
    std::vector<matmul_params> params(E);
    for (int e = 0; e < E; ++e) {
        srcs[e] = inp[e].get_raw_handle_unsafe();
        weis[e] = prepacked[e];
        dsts[e] = out[e].get_raw_handle_unsafe();
        params[e].dtypes.src = data_type_t::f16;
        params[e].dtypes.wei = data_type_t::f16;
        params[e].dtypes.dst = data_type_t::f16;
        params[e].dtypes.bias = data_type_t::none;
        params[e].mem_format_b = 'r';
        params[e].lowoha_algo = matmul_algo_t::moe_custom_kernel;
        params[e].num_threads = 0;
    }

    status_t st = group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks,
            alphas, srcs, ldas, weis, ldbs, biases, betas, dsts, ldcs, is_wc,
            params, /*moe_postop=*/nullptr);
    ASSERT_EQ(st, status_t::success)
            << "group_matmul_direct (f16, mem_format_b='r') failed";

    // No re-pack: a prepacked weight is consumed directly, so the eager
    // ALGO-3 warm must be skipped.
    auto stats = prepack::test_api::get_last_invocation_stats();
    EXPECT_FALSE(stats.valid)
            << "eager ALGO-3 warm ran for a prepacked (f16, mem_format_b='r') "
               "weight; it must be skipped";

    // Reference: per-expert GEMM on the ORIGINAL (un-packed) f16 weights.
    status_t ref_st = status_t::success;
    for (int e = 0; e < E && ref_st == status_t::success; ++e) {
        std::vector<post_op_type_t> ref_po;
        std::vector<tensor_t> bin;
        ref_st = matmul_kernel_test(inp[e], wt[e], bias[e], out_ref[e], ref_po,
                bin,
                /*use_LOWOHA=*/true, kAlgo,
                /*alpha=*/1.0f, /*beta=*/0.0f, /*use_reference=*/true);
    }
    ASSERT_EQ(ref_st, status_t::success) << "reference GEMM (f16) failed";

    bool ok = true;
    for (int e = 0; e < E && ok; ++e) {
        compare_tensor_2D_matrix(out[e], out_ref[e], M, N, K, f16_rtol, f16_eps,
                ok,
                /*enable_f32_relaxation=*/false, /*alpha=*/1.0f);
    }
    EXPECT_TRUE(ok) << "group_matmul f16 output mismatch vs reference "
                       "(prepacked weights)";

    for (int e = 0; e < E; ++e)
        zendnnl_aligned_free(prepacked[e]);
}

} // namespace
