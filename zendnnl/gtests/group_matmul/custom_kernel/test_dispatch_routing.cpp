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
#include <vector>

#include "ck_test_helpers.hpp"
#include "moe_test_utils.hpp"

namespace {

namespace mt = moe_test_utils;
namespace ck = ck_test::ck;
using mt::bfloat16_t;
using mt::data_type_t;
using mt::group_matmul_direct;
using mt::matmul_params;
using mt::status_t;

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
// `group_matmul/group_matmul_n_tile.cpp`), bypassing the custom
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
    mt::fill_src(src_bufs[e],  /*e=*/e);
    mt::fill_wei1(wei_bufs[e], /*e=*/e);
  }
  // Per-expert dst buffers, two passes (CK-fallback vs forced ALGO 1).
  std::vector<std::vector<bfloat16_t>> dst_ck(kNumOps);
  std::vector<std::vector<bfloat16_t>> dst_dlp(kNumOps);
  for (int e = 0; e < kNumOps; ++e) {
    dst_ck[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
    dst_dlp[e].assign(static_cast<size_t>(M) * N, bfloat16_t(0.0f));
  }

  std::vector<char>          layout(kNumOps, 'r');
  std::vector<bool>          transA(kNumOps, false), transB(kNumOps, false);
  std::vector<int>           Ms(kNumOps, M), Ns(kNumOps, N), Ks(kNumOps, K);
  std::vector<float>         alpha(kNumOps, 1.0f), beta(kNumOps, 0.0f);
  std::vector<int>           lda(kNumOps, K), ldb(kNumOps, N), ldc(kNumOps, N);
  std::vector<const void *>  src_ptrs(kNumOps), wei_ptrs(kNumOps);
  std::vector<const void *>  bias_ptrs(kNumOps, nullptr);
  for (int e = 0; e < kNumOps; ++e) {
    src_ptrs[e] = src_bufs[e].data();
    wei_ptrs[e] = wei_bufs[e].data();
  }
  std::vector<bool>          is_wc_false(kNumOps, false);

  std::vector<matmul_params> params(kNumOps);
  for (auto &p : params) {
    p.dtypes.src  = data_type_t::bf16;
    p.dtypes.wei  = data_type_t::bf16;
    p.dtypes.dst  = data_type_t::bf16;
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
    mt::AlgoEnvGuard            algo_guard(3);
    mt::CustomKernelOverride    ck_guard(true);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) dst_ptrs[e] = dst_ck[e].data();
    ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
                                  alpha, src_ptrs, lda, wei_ptrs, ldb,
                                  bias_ptrs, beta, dst_ptrs, ldc,
                                  is_wc_false, params,
                                  /*moe_postop=*/nullptr,
                                  /*gated_act=*/nullptr),
              status_t::success);
  }

  ::reset_grp_matmul_caches();
  // Run 2: ALGO=1 forced (always DLP) — reference for the comparison.
  {
    mt::AlgoEnvGuard algo_guard(1);
    std::vector<void *> dst_ptrs(kNumOps);
    for (int e = 0; e < kNumOps; ++e) dst_ptrs[e] = dst_dlp[e].data();
    ASSERT_EQ(group_matmul_direct(layout, transA, transB, Ms, Ns, Ks,
                                  alpha, src_ptrs, lda, wei_ptrs, ldb,
                                  bias_ptrs, beta, dst_ptrs, ldc,
                                  is_wc_false, params,
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
          << "CK-refused (fallback) output deviates from forced-ALGO-1 "
             "DLP output at expert=" << e << " i=" << i;
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
  EXPECT_EQ(
      ck::resolve_variant(data_type_t::bf16, data_type_t::bf16,
                           data_type_t::f16),
      ck::KernelVariant::kUnsupported);

  // prepare_for_call must also refuse on f16 dst, regardless of host ISA.
  if (ck::dispatch_supported()) {
    ck_test::PrepCallCase c{};
    c.dst_dt = data_type_t::f16;
    c.label = "f16_dst_refusal";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    EXPECT_EQ(ck_test::run_prepare(c, storage, kctx),
              status_t::failure);
    EXPECT_FALSE(kctx.enabled);
    EXPECT_EQ(kctx.variant, ck::KernelVariant::kUnsupported);
  }
}

// ──────────────────────────────────────────────────────────────────
// resolve_variant + prepare_for_call agree on rejected dtype tuples.
// The s8-bearing tuples below are not served by the kernel today;
// `resolve_variant` returns `kUnsupported` and `prepare_for_call`
// must refuse for the same inputs.  Locking the symmetry means a
// later change that adds new dtype rows updates both layers
// together.
// ──────────────────────────────────────────────────────────────────
TEST(CkDispatchRouting, ResolveAndPrepareAgreeOnRejectedTuples) {
  if (!ck::dispatch_supported()) {
    GTEST_SKIP() << "ISA gate would refuse anyway";
  }
  struct Row { data_type_t src, wei, dst; };
  for (auto r : {Row{data_type_t::bf16, data_type_t::s8,  data_type_t::bf16},
                 Row{data_type_t::bf16, data_type_t::s8,  data_type_t::f32 },
                 Row{data_type_t::s8 ,  data_type_t::s8,  data_type_t::bf16},
                 Row{data_type_t::s8 ,  data_type_t::s8,  data_type_t::f32 }}) {
    // resolve_variant rejects.
    EXPECT_EQ(ck::resolve_variant(r.src, r.wei, r.dst),
              ck::KernelVariant::kUnsupported)
        << "src="  << ck_test::dt_name(r.src)
        << " wei=" << ck_test::dt_name(r.wei)
        << " dst=" << ck_test::dt_name(r.dst)
        << " — if this row now resolves to a real variant, move it "
           "to test_resolve_variant.cpp's positive table.";
    // prepare_for_call rejects too.
    ck_test::PrepCallCase c{};
    c.src_dt = r.src;
    c.wei_dt = r.wei;
    c.dst_dt = r.dst;
    c.label = "rejected_dtype_tuple";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    EXPECT_EQ(ck_test::run_prepare(c, storage, kctx),
              status_t::failure);
    EXPECT_FALSE(kctx.enabled);
  }
}

}  // namespace
