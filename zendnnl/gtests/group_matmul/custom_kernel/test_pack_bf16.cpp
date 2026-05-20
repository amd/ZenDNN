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

/// CK pack module — `plan_pack_nr` boundary tests + cache-warming
/// invariants.  These tests exercise the pack module's public surface
/// (`plan_pack_nr` from `dispatch.hpp`, `warm_pack_all_custom_kernel_experts`
/// from `prepack/prepack_custom_kernel.hpp`) without going through the
/// per-tile dispatcher.
///
/// What's covered here:
///   * `plan_pack_nr` returns 32 / 64 / 0 per the documented contract.
///   * `prepare_for_call` implicitly chooses the right NR (validated
///     by checking `kctx.pack_nr` post-prepare).
///   * Cache-warm via `prepare_for_call` + a second `prepare_for_call`
///     hit — the second call must short-circuit the pack work
///     (verifiable by post-call `kctx.packed_ptrs` matching).
///
/// What's NOT covered here:
///   * Internal pack layout assertions (`[k_blocks, n_blocks, ...]`
///     ordering, K-pair stride correctness) — would require exposing
///     internal pack helpers or duplicating the layout in test code.
///     The end-to-end correctness in test_ukernel_bf16.cpp covers
///     this transitively.

#include <gtest/gtest.h>

#include "ck_test_helpers.hpp"

namespace {

namespace ck = ck_test::ck;
namespace mt = moe_test_utils;

// ──────────────────────────────────────────────────────────────────
// plan_pack_nr — pure function over (K, N).  Test the truth table.
// Today's contract:
//   * env override (ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR) wins when it
//     divides N; the test fixture below pins the override to 0
//     (= "auto / unset") via `mt::CustomKernelNROverride` so the
//     default 32 / 64 path is exercised deterministically — without
//     the override, the env getter's cached `static const` snapshot
//     could capture an externally set value (or a value cached by an
//     earlier test in the same process) and silently route the
//     truth-table assertions through the wrong branch.
//   * If N % 32 == 0, returns 32 (preferred — matches inner kernel's
//     register budget).
//   * Else if N % 64 == 0, returns 64.
//   * Else returns 0 (no supported NR divides N — caller refuses).
//
// Note: the second branch is only reachable if N % 32 != 0 AND
// N % 64 == 0, which is impossible mathematically (64 = 32×2), so
// the 64-branch is dead under default env.  The env override path
// CAN reach it; the `CkPlanPackNrOverride64` suite below pins
// `CustomKernelNROverride(64)` and asserts the alternate truth-table
// (N=64/128/192 → 64; N=32/96/100 → 0 because 64 doesn't divide).
// ──────────────────────────────────────────────────────────────────
struct PlanPackNrCase {
  int K, N;
  int expected;
  std::string label;
};

class CkPlanPackNrTest
    : public ::testing::TestWithParam<PlanPackNrCase> {
 protected:
  // Pin the NR knob to "auto / unset" for the duration of every
  // parameterised test instance so the default truth-table
  // assertions don't depend on the process's env state or cached
  // env snapshot.  Restored on test exit (RAII).
  mt::CustomKernelNROverride nr_guard{0};
};

TEST_P(CkPlanPackNrTest, MatchesContract) {
  const auto &c = GetParam();
  EXPECT_EQ(ck::plan_pack_nr(c.K, c.N), c.expected) << c.label;
}

INSTANTIATE_TEST_SUITE_P(
    Defaults, CkPlanPackNrTest,
    ::testing::Values(
        // Standard production shapes.
        PlanPackNrCase{64,    256, 32, "K64_N256"},
        PlanPackNrCase{2880, 5760, 32, "K2880_N5760"},
        PlanPackNrCase{2048, 1536, 32, "K2048_N1536"},
        PlanPackNrCase{4096, 14336, 32, "K4096_N14336_wide_N"},
        PlanPackNrCase{14336, 4096, 32, "K14336_N4096_tall_N"},
        // Smallest valid N.
        PlanPackNrCase{64,    32,  32, "smallest_N32"},
        PlanPackNrCase{1,     32,  32, "K1_N32"},  // K=1 still valid
        // N divisible by 32 — preferred.
        PlanPackNrCase{64,    96,  32, "N96"},
        PlanPackNrCase{64,    64,  32, "N64_div32"},  // div32 wins over div64
        PlanPackNrCase{64,   128,  32, "N128"},
        PlanPackNrCase{64,   160,  32, "N160_5x32"},
        // N indivisible by both 32 and 64 — refuse.
        PlanPackNrCase{64,    20,   0, "N20_neither"},
        PlanPackNrCase{64,    40,   0, "N40_8x5"},
        PlanPackNrCase{64,   100,   0, "N100"},
        PlanPackNrCase{64,   200,   0, "N200"},
        // Degenerate: N <= 0.
        PlanPackNrCase{64,     0,   0, "N0_degenerate"},
        PlanPackNrCase{64,    -1,   0, "N_negative"}),
    [](const ::testing::TestParamInfo<PlanPackNrCase> &info) {
      return info.param.label;
    });

// ──────────────────────────────────────────────────────────────────
// Override path: `CustomKernelNROverride(64)` pins the env-cached
// `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR` to 64.  Under that override,
// `plan_pack_nr` returns 64 when N % 64 == 0 and 0 otherwise — the
// default's preference for NR=32 is suppressed, so the 64-branch
// inside `plan_pack_nr` is the one being exercised.  Asserts the
// alternate truth-table the default suite above documents but does
// not itself reach (the default suite pins NR=0 = "auto").
// ──────────────────────────────────────────────────────────────────
class CkPlanPackNrOverride64Test
    : public ::testing::TestWithParam<PlanPackNrCase> {
 protected:
  mt::CustomKernelNROverride nr_guard{64};
};

TEST_P(CkPlanPackNrOverride64Test, MatchesOverrideContract) {
  const auto &c = GetParam();
  EXPECT_EQ(ck::plan_pack_nr(c.K, c.N), c.expected) << c.label;
}

INSTANTIATE_TEST_SUITE_P(
    Override64, CkPlanPackNrOverride64Test,
    ::testing::Values(
        // N % 64 == 0 — override returns 64.
        PlanPackNrCase{64,    64,  64, "ovr64_N64"},
        PlanPackNrCase{64,   128,  64, "ovr64_N128"},
        PlanPackNrCase{64,   192,  64, "ovr64_N192"},
        PlanPackNrCase{2048, 1536, 64, "ovr64_K2048_N1536"},  // Qwen narrow-N
        PlanPackNrCase{2880, 5760, 64, "ovr64_K2880_N5760"},
        PlanPackNrCase{4096, 14336, 64, "ovr64_K4096_N14336"},
        // N % 64 != 0 — override returns 0 (refused), even when
        // N % 32 == 0.  Demonstrates the override suppresses the
        // default's NR=32 fallback (i.e., the override is "force NR=64
        // or refuse"; it does NOT silently fall back to NR=32 for
        // shapes that don't divide).
        PlanPackNrCase{64,    32,   0, "ovr64_N32_indivisible_by_64"},
        PlanPackNrCase{64,    96,   0, "ovr64_N96_indivisible_by_64"},
        PlanPackNrCase{64,   160,   0, "ovr64_N160_indivisible_by_64"},
        // Indivisible by either — refused regardless of override.
        PlanPackNrCase{64,    20,   0, "ovr64_N20_neither"},
        PlanPackNrCase{64,   100,   0, "ovr64_N100"},
        // Degenerate.
        PlanPackNrCase{64,     0,   0, "ovr64_N0_degenerate"},
        PlanPackNrCase{64,    -1,   0, "ovr64_N_negative"}),
    [](const ::testing::TestParamInfo<PlanPackNrCase> &info) {
      return info.param.label;
    });

// ──────────────────────────────────────────────────────────────────
// `prepare_for_call`-side: post-prepare `kctx.pack_nr` matches what
// `plan_pack_nr` reports for the same (K, N).
// ──────────────────────────────────────────────────────────────────
TEST(CkPackBf16, PrepareReportsPlanPackNr) {
  CK_SKIP_IF_NO_BF16_ISA();
  // Pin NR to "auto / unset" for the same reason as the parameterised
  // suite above: the assertion is "kctx.pack_nr matches plan_pack_nr",
  // which holds for any NR setting, but pinning makes the test
  // self-contained and immune to externally cached env state.
  mt::CustomKernelNROverride nr_guard(0);
  for (int N : {32, 64, 96, 128, 256, 512, 1024, 1536, 2880, 5760}) {
    ck_test::PrepCallCase c{};
    c.N = N;
    c.label = std::string("ppnr_N") + std::to_string(N);
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    const auto status = ck_test::run_prepare(c, storage, kctx);
    ASSERT_EQ(status, zendnnl::error_handling::status_t::success)
        << "case=" << c.label;
    EXPECT_EQ(kctx.pack_nr, ck::plan_pack_nr(c.K, c.N))
        << "kctx.pack_nr deviates from plan_pack_nr — case=" << c.label;
  }
}

// ──────────────────────────────────────────────────────────────────
// Cache-warm symmetry: the same weight pointer + (K, N, ldb)
// produces the same packed pointer across two `prepare_for_call`
// invocations (the LRU pack cache hits on the second call).  This
// also incidentally verifies that the dispatcher does not zero-init
// `kctx.packed_ptrs` to nullptr on the second call.
// ──────────────────────────────────────────────────────────────────
TEST(CkPackBf16, SecondPrepareHitsCacheSamePackedPtr) {
  CK_SKIP_IF_NO_BF16_ISA();
  // Reset the process-wide pack cache before this test runs — the
  // assertion below ("same weight pointer + shape → same packed
  // pointer across two prepare calls") only holds when the cache
  // starts empty.  Without the reset, an earlier test in the same
  // process can leak a packed entry whose key happens to collide
  // with the local `PrepCallStorage`'s weight pointer (the
  // allocator routinely reuses heap addresses when stack-local
  // storage of the same size goes out of scope), yielding a
  // first-prepare cache HIT against a STALE entry that does not
  // correspond to the current weight bytes — the second prepare
  // would then either match (false pass) or differ (false fail)
  // for reasons unrelated to the LRU contract this test pins.
  ::reset_grp_matmul_caches();

  ck_test::PrepCallCase c{};
  c.label = "cache_warm_symmetry";

  ck_test::PrepCallStorage storage;
  ck::CallContext kctx_a, kctx_b;
  ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success);
  ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_b),
            zendnnl::error_handling::status_t::success);

  // Same weight buffer + same shape = same packed pointer in the LRU.
  EXPECT_NE(kctx_a.packed_ptrs[0], nullptr);
  EXPECT_EQ(kctx_a.packed_ptrs[0], kctx_b.packed_ptrs[0])
      << "cache-warm symmetry broken: same (weight ptr, K, N, ldb) "
         "produced different packed pointers across calls — the LRU "
         "pack cache likely has a key inconsistency";
}

// ──────────────────────────────────────────────────────────────────
// Different weight pointer → different packed pointer.  Ensures the
// LRU is keyed on the source pointer (not just (K, N)).  Important
// because in production each expert has its own weight buffer.
// ──────────────────────────────────────────────────────────────────
TEST(CkPackBf16, DistinctWeightPointersProduceDistinctPacks) {
  CK_SKIP_IF_NO_BF16_ISA();
  ck_test::PrepCallCase c{};

  // Run two prepares with disjoint weight buffers.
  ck_test::PrepCallStorage s1, s2;
  ck::CallContext kctx1, kctx2;
  ASSERT_EQ(ck_test::run_prepare(c, s1, kctx1),
            zendnnl::error_handling::status_t::success);
  ASSERT_EQ(ck_test::run_prepare(c, s2, kctx2),
            zendnnl::error_handling::status_t::success);

  EXPECT_NE(kctx1.packed_ptrs[0], nullptr);
  EXPECT_NE(kctx2.packed_ptrs[0], nullptr);
  EXPECT_NE(kctx1.packed_ptrs[0], kctx2.packed_ptrs[0])
      << "two distinct weight buffers produced the same packed "
         "pointer — LRU is not keyed on weight pointer";
}

}  // namespace
