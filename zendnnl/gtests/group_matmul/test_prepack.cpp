/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file test_group_matmul_prepack.cpp
/// @brief Gtest sections covering the `group_matmul/prepack/` module —
///        per-ALGO prepack functions, custom-kernel + AOCL DLP backend
///        warmers, and end-to-end fused-MoE prepack invariants.
///
/// Split from `test_group_matmul.cpp` (Phase A refactor) so prepack-
/// module tests live in their own translation unit; the parent file
/// stays focused on the basic group_matmul / fused-MoE / quant suites.
/// Both files share helpers via `moe_test_utils.hpp`.
///
///   [16] TestFusedMoECacheStress           — direct cache-cold/warm
///                                             on warm_pack_all_custom_kernel
///   [17] TestFusedMoEPointerChurn          — cache keyed on weight ptrs
///   [18] TestPrepackPerAlgoFunctions       — per-ALGO prepack surface
///                                             (incl. per-tile AOCL DLP)
///   [19] TestPrepackFusedMoEEndToEnd       — Pass 2 K_down sizing + propagation
///   [20] TestPrepackKDownSynthesisAllActs  — K_down across every gated_act
///   [21] TestPrepackResultInvariance       — cold→warm + ALGO 1↔3 invariance
///   [22] TestPrepackClearCacheDirect       — cache-clear regression
///   [23] TestPrepackAoclDlpFullWeight      — full-weight is_weights_const gate
///   [24] TestPrepackVariableNExperts       — per-expert N skew
///   [25] TestPrepackStressManyExperts      — E=64 multi-iter + E=256 boundary
///   [26]-[28] TestPrepackEnv* (Bucket A/B + Interaction matrix)
///                                          — subprocess-isolated env-knob
///                                            coverage (gtest threadsafe
///                                            death-test pattern)
///   [30] TestPrepackFingerprintInvariance  — order-independent fingerprint
///                                            (permutation/membership/pool-size)
///   [32] TestPrepackCkGateSymmetry         — prepack mirrors runtime CK refusal
///                                            gates (act, act_dtype, bias_dtype,
///                                            pack_nr)
///   [33] TestPrepackBuildParamsContract    — build_prepack_params honours the
///                                            dispatcher active/total contract

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

// Shared MoE / fused-MoE / group_matmul gtest helpers.
#include "moe_test_utils.hpp"

// Direct access to the custom-kernel warm-pack helper for the cache-
// stress tests — those need the per-call HIT/MISS counts from
// `PackProbeStats` to assert the warm-pack actually populates the
// LRU on miss and serves on hit.  After the prepack-module refactor,
// the warmer + stats live under
// `group_matmul/prepack/prepack_custom_kernel.hpp`; the per-call
// dispatcher header (`custom_kernel/dispatch.hpp`) is still pulled in
// for completeness, and `pack.hpp` is needed for
// `clear_custom_kernel_pack_cache()`.
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_aocl_dlp.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack_custom_kernel.hpp"

// ===============================================================================
// [16] TestFusedMoECacheStress - calls warm_pack_all_custom_kernel_experts
//      directly (not through group_matmul_direct) on a 64-expert bank,
//      then re-warms to assert every expert is now a HIT.  Direct-API
//      surface check that complements [12] (which exercises the
//      warm-pack hook through the dispatcher).
// ===============================================================================

class TestFusedMoECacheStress : public ::testing::Test {};

TEST_F(TestFusedMoECacheStress, ColdThenWarm) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  // Wipe every group_matmul-related cache (CK pack arena, prepack
  // fingerprint, AOCL/oneDNN/native weight LRUs) up front so this
  // test is independent of test-ordering effects.  The AOCL LRU is
  // process-wide and pointer-keyed; without this clear, a TypedBuffer
  // freed by an earlier test whose heap address is reused here would
  // stale-hit and return another test's reordered weights.  Safe
  // outside any OMP region (single-threaded gtest main thread).
  reset_grp_matmul_caches();

  // Cold bank of 64 experts (small enough to keep the test fast,
  // large enough to be representative).  N=64 is a multiple of 32
  // so the custom-kernel pack_nr planner picks NR=32.
  constexpr int E = 64;
  const int K = 32, N = 64, ldb = N;
  TypedBuffers wei_buf;
  wei_buf.alloc(E, (size_t)K * N, /*is_bf16=*/true);
  fill_moe_tensors(E, /*is_bf16=*/true, nullptr, &wei_buf, nullptr);
  auto wei = wei_buf.cptrs(true);
  std::vector<int>  K_vec(E, K), N_vec(E, N), ldb_vec(E, ldb);
  std::vector<bool> transB(E, false);

  // First warm-up: every expert MUST miss (cache is freshly cleared).
  group_matmul_prepack::custom_kernel::PackProbeStats st_cold;
  ASSERT_EQ(group_matmul_prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
                wei, K_vec, N_vec, ldb_vec, transB,
                /*is_weights_const=*/std::vector<bool>{}, E, st_cold),
            status_t::success);
  EXPECT_EQ(st_cold.total_attempted, E);
  EXPECT_EQ(st_cold.cache_misses,    E);
  EXPECT_EQ(st_cold.cache_hits,       0);
  EXPECT_EQ(st_cold.skipped_invalid,  0);
  EXPECT_EQ(st_cold.packed_ok,       E);

  // Second warm-up on identical inputs: every expert MUST hit.
  group_matmul_prepack::custom_kernel::PackProbeStats st_warm;
  ASSERT_EQ(group_matmul_prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
                wei, K_vec, N_vec, ldb_vec, transB,
                /*is_weights_const=*/std::vector<bool>{}, E, st_warm),
            status_t::success);
  EXPECT_EQ(st_warm.total_attempted, E);
  EXPECT_EQ(st_warm.cache_hits,      E);
  EXPECT_EQ(st_warm.cache_misses,     0);
  EXPECT_EQ(st_warm.skipped_invalid,  0);
  EXPECT_EQ(st_warm.packed_ok,       E);
}

// ===============================================================================
// [17] TestFusedMoEPointerChurn - validates the pack cache is keyed on
//      weight pointer (not per-expert index).  Re-allocating one
//      expert's weight (different .data()) MUST cause exactly that one
//      expert to miss; the rest stay hits.
// ===============================================================================

class TestFusedMoEPointerChurn : public ::testing::Test {};

TEST_F(TestFusedMoEPointerChurn, ReallocOneExpertOnlyMisses) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;

  // Pointer-churn test: any stale AOCL LRU entry from an earlier
  // test (same K/N/ldb, freed-then-reused weight ptr) would mask the
  // miss-vs-hit accounting we're measuring here.  Wipe everything.
  reset_grp_matmul_caches();

  constexpr int E = 8;
  const int K = 32, N = 64, ldb = N;
  TypedBuffers wei_buf;
  wei_buf.alloc(E, (size_t)K * N, /*is_bf16=*/true);
  fill_moe_tensors(E, /*is_bf16=*/true, nullptr, &wei_buf, nullptr);
  auto wei = wei_buf.cptrs(true);
  std::vector<int>  K_vec(E, K), N_vec(E, N), ldb_vec(E, ldb);
  std::vector<bool> transB(E, false);

  // Cold warm-pack -> all miss.
  group_matmul_prepack::custom_kernel::PackProbeStats st1;
  ASSERT_EQ(group_matmul_prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
                wei, K_vec, N_vec, ldb_vec, transB,
                /*is_weights_const=*/std::vector<bool>{}, E, st1),
            status_t::success);
  ASSERT_EQ(st1.cache_misses, E);

  // Reallocate expert 3's weight buffer and refill with the same seed.
  // The new buffer has a different .data() pointer, so the cache key
  // (which folds in the pointer) is distinct.
  std::vector<bfloat16_t> realloc_buf((size_t)K * N);
  fill_wei1(realloc_buf, /*seed=*/3);
  wei[3] = realloc_buf.data();

  // Re-warm: 7 hits + 1 miss (the reallocated one).
  group_matmul_prepack::custom_kernel::PackProbeStats st2;
  ASSERT_EQ(group_matmul_prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
                wei, K_vec, N_vec, ldb_vec, transB,
                /*is_weights_const=*/std::vector<bool>{}, E, st2),
            status_t::success);
  EXPECT_EQ(st2.cache_hits,   E - 1);
  EXPECT_EQ(st2.cache_misses,     1);
  EXPECT_EQ(st2.packed_ok,       E);
}

// ===============================================================================
// [18] TestPrepackPerAlgoFunctions - drive the per-ALGO prepack functions
//      directly through their public API (group_matmul/prepack/prepack.hpp)
//      without going through group_matmul_direct.  Asserts the four shared
//      short-circuits (env knob OFF, total<=active, fingerprint hit,
//      inner != aocl_dlp_blocked) plus the ALGO 3 custom-kernel branch.
// ===============================================================================

class TestPrepackPerAlgoFunctions : public ::testing::Test {};

namespace {
// Build a minimal `PrepackParams` that points at caller-owned vectors.
// Each "expert" gets its own owned buffer so each cache key
// (weight_ptr, K, N, ldb, transB, pack_nr) is unique — necessary for
// the post-prepack probe to give meaningful hit/miss counts (a
// shared buffer collapses all entries into one cache key, so the
// probe's own warm-on-miss behaviour pollutes the result).
struct PrepackHarness {
  std::vector<std::vector<bfloat16_t>> banks;     // per-expert weight storage
  std::vector<const void *> weight;
  std::vector<int>          K;
  std::vector<int>          N;
  std::vector<int>          ldb;
  std::vector<bool>         transB;
  std::vector<bool>         is_weights_const;
  zendnnl::lowoha::matmul::group_matmul_prepack::PrepackParams pp;
};

PrepackHarness make_harness(int total, int active, int K_v, int N_v,
                            float fill_value) {
  using namespace zendnnl::common;
  PrepackHarness h;
  h.banks.resize(total);
  h.weight.resize(total);
  h.K.assign(total, K_v);
  h.N.assign(total, N_v);
  h.ldb.assign(total, N_v);
  h.transB.assign(total, false);
  h.is_weights_const.assign(total, true);
  for (int e = 0; e < total; ++e) {
    h.banks[e].assign((size_t)K_v * N_v,
                      bfloat16_t(fill_value + 0.001f * (float)e));
    h.weight[e] = h.banks[e].data();
  }
  h.pp.weight           = &h.weight;
  h.pp.K                = &h.K;
  h.pp.N                = &h.N;
  h.pp.ldb              = &h.ldb;
  h.pp.transB           = &h.transB;
  h.pp.is_weights_const = &h.is_weights_const;
  h.pp.src_dtype = data_type_t::bf16;
  h.pp.wei_dtype = data_type_t::bf16;
  h.pp.dst_dtype = data_type_t::bf16;
  h.pp.num_ops_active = active;
  h.pp.num_ops_total  = total;
  h.pp.custom_kernel_on = true;
  return h;
}
} // namespace

TEST_F(TestPrepackPerAlgoFunctions, EagerWarmsAllExpertsWhenTotalEqualsActive) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Under the uniform-eager semantic (PR #443's prepack module), the
  // per-ALGO functions always warm the firing experts when
  // PREPACK=ON, regardless of whether the framework opted into the
  // `total_matmul > active_matmul` contract.  This test pins that
  // behaviour: with `total == active == 4`, ALGO 3 + custom-kernel
  // (the only path with observable cache state via PackProbeStats)
  // must populate all 4 entries.
  //
  // The prior gtest (`EarlyOutTotalEqualsActive`) asserted the
  // OPPOSITE invariant — that prepack short-circuited when
  // total == active.  That gate was deliberately removed; if a
  // future refactor reintroduces it, this test fires.
  //
  // PREPACK=0 escape hatch is exercised separately via the
  // subprocess env-matrix tests in section [26].
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.5f);
  prepack::prepack_for_algo_1(h.pp);
  prepack::prepack_for_algo_2(h.pp);
  prepack::prepack_for_algo_3(h.pp);
  prepack::prepack_for_algo_4(h.pp);
  prepack::prepack_for_algo_5(h.pp);

  // Probe the custom-kernel cache.  ALGO 3's body warms it (BF16 +
  // custom-kernel-on satisfied via `make_harness`); the other ALGO
  // bodies don't touch the custom cache, so the probe reflects the
  // single contribution from `prepack_for_algo_3`.
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits, 4)
      << "expected eager pre-warming of all 4 experts when "
         "total==active under PREPACK=ON; if cache_hits<4 the "
         "uniform-eager prelude has regressed (the `total<=active` "
         "short-circuit is back)";
  EXPECT_EQ(probe.cache_misses, 0);
}

TEST_F(TestPrepackPerAlgoFunctions, FingerprintCacheDedup) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.25f);

  // First call on ALGO 3: warm-pack should populate the BF16 custom
  // kernel cache for all 4 entries (custom_kernel_on=true,
  // BF16/BF16/BF16, total > active).  AOCL DLP cache is also warmed
  // when inner_kernel resolves to aocl_dlp_blocked, but we check the
  // observable side (custom-kernel HIT/MISS) below.
  prepack::prepack_for_algo_3(h.pp);

  // Second call with the same fingerprint: should short-circuit via
  // the per-thread fingerprint cache, no new packs.
  prepack::prepack_for_algo_3(h.pp);

  // Probe the cache state via warm_pack_all_custom_kernel_experts:
  // since the first call populated all 4 entries and the second was
  // a no-op, we should now see all 4 as HITS.
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits,   4)
      << "expected the first prepack_for_algo_3 to have populated all "
         "entries; the second invocation must have been a fingerprint "
         "no-op (otherwise cache would also be warm).";
  EXPECT_EQ(probe.cache_misses, 0);
}

TEST_F(TestPrepackPerAlgoFunctions, Algo3CustomKernelBranch) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Use a different fill value than the previous test so the
  // fingerprint differs and the warm-pack actually fires.
  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.75f);

  prepack::prepack_for_algo_3(h.pp);

  // Custom-kernel branch fired (BF16 + custom_kernel_on); a probe
  // should now hit all 4 entries.
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits, 4) << "ALGO 3 + BF16 + custom_kernel_on must "
                                    "have warmed the custom-kernel cache";
}

TEST_F(TestPrepackPerAlgoFunctions, Algo3CustomKernelBranchSkippedOnNonBF16) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Different weights again to dodge the per-thread fingerprint
  // cache from previous tests in this binary.
  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.125f);
  // Flip src dtype to F32 — the eligibility predicate now demands
  // ALL of (custom_kernel_on, src=BF16, wei=BF16, dst=BF16); failing
  // any one disables the custom-kernel branch.  AOCL is also gated
  // on BF16-only in `prepack_aocl_dlp.cpp`, so neither warmer fires.
  h.pp.src_dtype = data_type_t::f32;

  prepack::prepack_for_algo_3(h.pp);

  // Cache should be UNTOUCHED — both the custom-kernel and AOCL DLP
  // backends gate on BF16 weight dtype, and the dispatcher's
  // `resolve_kernel()` returns aocl_dlp_blocked but the AOCL warmer
  // would skip these BF16 weights anyway because the wei_dtype
  // metadata signals non-BF16 to the helper.  Verify by re-warming
  // and asserting all-misses.
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_misses, 4)
      << "non-BF16 src dtype must skip the custom-kernel branch";
}

// ── AOCL DLP per-tile warm-pack coverage ─────────────────────────────
//
// The `warm_pack_all_aocl_dlp_experts_n_tile(...)` warmer mirrors the
// runtime `aligned_n_split(N[e], n_thr_e, tid, nr_align)` decomposition
// used by ALGO 3 flat_n_tile under the strict-stable plan
// (`ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE=1`, default).  These tests
// drive the warmer directly so we can assert byte-precise pack counts
// independent of the global env-knob caching that
// `prepack_for_algo_3(...)` is subject to.
//
// Counts asserted: `total_attempted = sum_e min(stable, N[e]/nr_align)`
// for valid experts; `packed_ok` ≤ `total_attempted` (the AOCL DLP
// reorder primitive may report a per-tile failure on degenerate input,
// counted as `skipped_invalid`).  We use uniform-N inputs so the
// per-expert clamp `min(stable, N[e]/nr_align)` collapses to a single
// constant.

TEST_F(TestPrepackPerAlgoFunctions, AoclDlpNTileTotalAttemptedCount) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  // Per-tile warmer is independent of the custom-kernel cache, but
  // the fingerprint cache is per-thread so we still clear it for
  // isolation from neighbouring tests.
  reset_grp_matmul_caches();

  // total=4, K=64, N=128 (large enough that the AOCL DLP reorder
  // primitive accepts the slice).  num_threads=64 → stable = 64/16 = 4.
  // align_cap = N / nr_align = 128 / 1 = 128 ≥ stable, so
  // n_thr_e = 4 for every expert and total_attempted = 4 × 4 = 16.
  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.5f);

  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4,
          /*wei_dtype=*/data_type_t::bf16,
          /*num_threads=*/64,
          /*stable=*/4,
          /*nr_align=*/1,
          st),
      status_t::success);

  // 4 experts × 4 tiles = 16 attempts, each producing a per-tile
  // cache key.  packed_ok + skipped_invalid must equal total_attempted.
  EXPECT_EQ(st.total_attempted, 16);
  EXPECT_EQ(st.packed_ok + st.skipped_invalid, st.total_attempted);
  // Production-cache gate must be ON for the BF16 reorder primitive
  // to accept the request; if `ZENDNNL_MATMUL_WEIGHT_CACHE != 1` the
  // warmer short-circuits with all counters at zero.  In that mode
  // the assertion below is informational rather than load-bearing.
  if (st.total_attempted > 0) {
    EXPECT_GT(st.packed_ok, 0)
        << "weight-cache type 1 (default) should accept BF16 reorders";
  }
}

TEST_F(TestPrepackPerAlgoFunctions, AoclDlpNTilePerExpertNarrowNClamp) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // num_threads=64 → stable=4; nr_align=16; N=64
  // → align_cap = 64/16 = 4; n_thr_e = min(4, 4) = 4 → 4 tiles.
  auto h = make_harness(/*total=*/3, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.25f);

  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/3,
          /*wei_dtype=*/data_type_t::bf16,
          /*num_threads=*/64,
          /*stable=*/4,
          /*nr_align=*/16,
          st),
      status_t::success);
  EXPECT_EQ(st.total_attempted, 12);  // 3 × min(4, 64/16) = 3 × 4
  EXPECT_EQ(st.packed_ok + st.skipped_invalid, st.total_attempted);
}

TEST_F(TestPrepackPerAlgoFunctions, AoclDlpNTileNonBf16IsSkipped) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.5f);

  // Force F32 — the warmer mirrors `run_dlp(...)`'s BF16-only gate
  // (other dtypes count every entry as `skipped_invalid` and exit
  // without touching the cache, see `prepack_aocl_dlp.cpp`).
  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4,
          /*wei_dtype=*/data_type_t::f32,
          /*num_threads=*/64,
          /*stable=*/4,
          /*nr_align=*/1,
          st),
      status_t::success);

  // Non-BF16 path: every reachable entry counts as `skipped_invalid`,
  // none packs.  `total_attempted` matches `skipped_invalid` (the
  // dtype-skip path increments both by the per-expert reachable bound
  // for cross-regime accounting parity; see the warmer's dtype-skip
  // block doc-comment).  Bound = min(total_count, weight.size(),
  // K.size(), N.size(), ldb.size(), transB.size()) = 4 here.
  EXPECT_EQ(st.packed_ok, 0);
  EXPECT_EQ(st.total_attempted, 4);
  EXPECT_EQ(st.skipped_invalid, 4);
  EXPECT_EQ(st.packed_ok + st.skipped_invalid, st.total_attempted);
}

TEST_F(TestPrepackPerAlgoFunctions, AoclDlpNTileSkipsNonConstExperts) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.75f);
  // Mark expert 1 + 3 as variable-weight — `run_dlp(...)` won't cache
  // those at runtime, so the warmer must skip them too (mirroring
  // the runtime gate at aocl_kernel.cpp:1700-1702).
  h.is_weights_const[1] = false;
  h.is_weights_const[3] = false;

  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts_n_tile(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4,
          /*wei_dtype=*/data_type_t::bf16,
          /*num_threads=*/64,
          /*stable=*/4,
          /*nr_align=*/1,
          st),
      status_t::success);

  // Two valid const experts × 4 tiles = 8 attempts.  The two
  // non-const experts are counted as `skipped_invalid` (one count
  // each, NOT four — the warmer skips at expert granularity for the
  // is_weights_const gate, mirroring `run_dlp`'s coarser gate).
  EXPECT_EQ(st.total_attempted, 8 + 2);
  EXPECT_EQ(st.skipped_invalid, 2);
  EXPECT_EQ(st.packed_ok + (st.skipped_invalid - 2), 8);
}

TEST_F(TestPrepackPerAlgoFunctions, CustomKernelWarmerSkipsNonConstExperts) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Targeted regression for Bug-class B-CK-IWC-1 (Copilot review):
  // the custom-kernel warm-pack used to ignore `is_weights_const`
  // entirely, populating the CK pack cache for every reachable
  // expert.  When the caller mutated a non-const weight buffer
  // in-place between calls, subsequent CK dispatches would silently
  // serve the stale cached pack.  Fix is symmetric with the AOCL
  // DLP warmer (see `AoclDlpNTileSkipsNonConstExperts` above): skip
  // packing for any expert where `is_weights_const[i] == false`.
  // The CK runtime separately refuses any call containing a non-
  // const active expert (see
  // `custom_kernel/dispatch.cpp::prepare_for_call`), so a non-const
  // expert never benefits from warming on either side.
  //
  // Shape: K=64 (multiple of 4 VNNI pair count), N=128 (multiple
  // of pack_nr=64 — keeps `plan_pack_nr` happy).
  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.5f);
  h.is_weights_const[1] = false;
  h.is_weights_const[3] = false;

  prepack::custom_kernel::PackProbeStats st;
  ASSERT_EQ(
      prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4, st),
      status_t::success);

  // Four entries reach the warmer; experts [1, 3] are skipped via
  // the new is_weights_const gate.  Experts [0, 2] pack on cold
  // cache (cache_misses += 2), and `packed_ok == 2`.
  EXPECT_EQ(st.total_attempted, 4);
  EXPECT_EQ(st.skipped_invalid, 2)
      << "Two non-const experts must be skipped (mirrors the AOCL "
         "DLP warmer's gate at `prepack_aocl_dlp.cpp::warm_pack_all_"
         "aocl_dlp_experts`)";
  EXPECT_EQ(st.packed_ok,    2);
  EXPECT_EQ(st.cache_hits,   0);
  EXPECT_EQ(st.cache_misses, 2);

  // Re-running with the same is_weights_const must HIT for the two
  // const experts (cache populated above) and skip the same two
  // non-const ones.  Validates idempotent skip + cache reuse.
  prepack::custom_kernel::PackProbeStats st2;
  ASSERT_EQ(
      prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4, st2),
      status_t::success);
  EXPECT_EQ(st2.cache_hits,   2);
  EXPECT_EQ(st2.cache_misses, 0);
  EXPECT_EQ(st2.skipped_invalid, 2);

  // Empty `is_weights_const` is the documented "treat as const"
  // sentinel.  Same call but with the flag vector cleared must
  // warm all four experts (the two we already cached above HIT,
  // the two we previously skipped MISS now that they're allowed).
  prepack::custom_kernel::PackProbeStats st3;
  static const std::vector<bool> kEmpty;
  ASSERT_EQ(
      prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
          h.weight, h.K, h.N, h.ldb, h.transB, kEmpty,
          /*total_count=*/4, st3),
      status_t::success);
  EXPECT_EQ(st3.skipped_invalid, 0)
      << "Empty is_weights_const must NOT skip any expert "
         "(legacy `treat as const` sentinel)";
  EXPECT_EQ(st3.cache_hits,   2)
      << "Experts 0 + 2 already warmed in earlier calls";
  EXPECT_EQ(st3.cache_misses, 2)
      << "Experts 1 + 3 (previously skipped) now pack and miss";
}

TEST_F(TestPrepackPerAlgoFunctions, Algo3PerTilePathRunsWithThreadContext) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // End-to-end through `prepack_for_algo_3` with a thread context
  // populated.  This is the path `flat_n_tile` takes in production.
  // We can't directly observe the AOCL DLP cache (the per-dtype LRU
  // is private to `aocl_kernel.cpp`), but we can assert that the
  // custom-kernel side still warms — the dispatcher's per-tile AOCL
  // branch must NOT short-circuit the custom-kernel branch.
  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.625f);
  h.pp.num_threads = 64;
  h.pp.nr_align    = 1;

  prepack::prepack_for_algo_3(h.pp);

  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits,   4)
      << "ALGO 3 + BF16 + custom_kernel_on must warm the custom-kernel "
         "cache regardless of the AOCL DLP path taken";
  EXPECT_EQ(probe.cache_misses, 0);
}

// ===============================================================================
// [19] TestPrepackFusedMoEEndToEnd - regression test for the Pass 2
//      K_down sizing bug.  After ONE group_matmul_direct call with
//      fused_moe set, active=K, total=E (where E > K), every one of
//      the E experts' Op1 weights AND Op2 weights must be in the
//      custom-kernel cache (cache_hits = E for both probes).
//
//      The fused-MoE flow internally invokes
//      group_matmul_run_parallel_dispatch TWICE (Pass 1 with Op1
//      weights, Pass 2 with Op2's down_weight); each pass triggers
//      its per-ALGO prepack which warms its own weights.  The
//      original Pass-2 implementation under-warmed Op2 because
//      `scratch.K_down` was sized to `num_ops` (active) instead of
//      `N.size()` (total) - the AOCL/custom-kernel warmer's min()
//      clamp would truncate to `num_ops`, leaving the
//      [num_ops, num_ops_total) tail of Op2 weights cold.
//      Existing pipeline tests didn't catch this because they only
//      assert numerical correctness (which is preserved by on-demand
//      packing); a direct probe of cache state is the only way to
//      surface the regression.
// ===============================================================================

class TestPrepackFusedMoEEndToEnd : public ::testing::Test {};

TEST_F(TestPrepackFusedMoEEndToEnd, BothPassesWarmAllExperts) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Force ALGO=3 + custom-kernel ON so the prepack-for-algo-3 body
  // fires both AOCL and custom-kernel branches (the only path with
  // observable HIT/MISS counters is the custom-kernel side).
  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  constexpr int E    = 8;          // total expert count
  constexpr int K    = 4;          // active fired count
  const int     dim  = 32;
  const int     N1   = 2 * dim;    // Op1 N (gate+up)
  const int     H    = 32;
  const int     M    = 4;          // tokens per fired expert
  const int     K_in = H;
  const auto    act  = grp_matmul_gated_act_t::silu_and_mul;
  const int     K_op2 = N1 / 2;     // op2_k_for_act(N1, silu) = N1/2
  const data_type_t dt = data_type_t::bf16;

  // Per-expert weights with distinct buffers so each cache key is
  // unique (the probe's own warm-on-miss can't pollute the count).
  TypedBuffers w1_all, w2_all;
  w1_all.alloc(E, (size_t)K_in * N1,    /*is_bf16=*/true);
  w2_all.alloc(E, (size_t)K_op2 * H,    /*is_bf16=*/true);
  for (int e = 0; e < E; ++e) {
    fill_wei1(w1_all.bf16[e], e);
    fill_wei2(w2_all.bf16[e], e);
  }
  auto wei1_all = w1_all.cptrs(true);
  auto wei2_all = w2_all.cptrs(true);

  // Active prefix [0, K) gets meaningful src; tail [K, E) is M=0.
  TypedBuffers src_test;
  src_test.alloc(E, (size_t)M * K_in, true);
  for (int e = 0; e < K; ++e) fill_src(src_test.bf16[e], e);
  auto src_test_p = src_test.cptrs(true);

  auto gv = GemmVecs::uniform(E, M, N1, K_in);
  for (int e = K; e < E; ++e) gv.Ms[e] = 0;

  std::vector<const void *> no_bias(E, nullptr);
  auto params = make_uniform_params(E, dt);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(K);
    pp.total_matmul  = static_cast<uint32_t>(E);
  }

  grp_matmul_gated_act_params act_params{};
  act_params.act = act;

  auto fused = make_fused_moe_op2(E, H, wei2_all, no_bias);

  std::vector<void *> dst_null(E, nullptr);
  std::vector<int>    ldc_null(E, 0);
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                src_test_p, gv.lda, wei1_all, gv.ldb,
                                no_bias, gv.beta, dst_null, ldc_null,
                                gv.is_wc, params, nullptr, &act_params,
                                &fused),
            status_t::success);

  // ── Probe Op1 ──────────────────────────────────────────────────
  // After the call, ALL E Op1 weights should be in the custom-kernel
  // cache (Pass 1 prepack warmed them).
  std::vector<int>  K_op1_vec(E, K_in);
  std::vector<int>  N_op1_vec(E, N1);
  std::vector<int>  ldb_op1(E, N1);
  std::vector<bool> transB_op1(E, false);
  prepack::custom_kernel::PackProbeStats probe_op1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei1_all, K_op1_vec, N_op1_vec, ldb_op1, transB_op1,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op1);
  EXPECT_EQ(probe_op1.cache_hits, E)
      << "all " << E << " Op1 weights must be in the custom-kernel cache "
      << "after Pass 1's prepack_for_algo_3 fired";
  EXPECT_EQ(probe_op1.cache_misses, 0);

  // ── Probe Op2 ──────────────────────────────────────────────────
  // After the call, ALL E Op2 weights should ALSO be in the custom-
  // kernel cache (Pass 2 prepack warmed them).  This is the
  // assertion that catches the scratch.K_down sizing bug: a broken
  // implementation would only have the [0, K) prefix cached and
  // would miss on entries [K, E).
  std::vector<int>  K_op2_vec(E, K_op2);
  std::vector<int>  N_op2_vec(E, H);
  std::vector<int>  ldb_op2(E, H);
  std::vector<bool> transB_op2(E, false);
  prepack::custom_kernel::PackProbeStats probe_op2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei2_all, K_op2_vec, N_op2_vec, ldb_op2, transB_op2,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op2);
  EXPECT_EQ(probe_op2.cache_hits, E)
      << "all " << E << " Op2 weights must be in the custom-kernel cache "
      << "after Pass 2's prepack_for_algo_3 fired - if cache_misses > 0 "
      << "for entries [" << K << ", " << E << "), Pass 2's scratch.K_down "
      << "is back to being sized to `num_ops` (active) instead of "
      << "`N.size()` (total), and the [active, total) prepack-extras "
      << "tail of Op2 weights is no longer being warmed.";
  EXPECT_EQ(probe_op2.cache_misses, 0);
}

// ===============================================================================
// [20] TestPrepackKDownSynthesisAllActs - parameterised regression test
//      for `op2_k_for_act()` correctness across every gated_act value.
//
//      The historical BLOCKER bug had `K_down_synth` derive its half-K
//      flag from `swiglu_oai_mul` only — `silu_and_mul` and
//      `gelu_and_mul` slipped through and warm-packed Op2 with K=N1
//      while runtime later looked up under K=N1/2.  The single-act E2E
//      test [19] was fixed for `silu_and_mul` but a future refactor
//      could re-introduce the regression for any of the four enum
//      values, so we sweep all of them here.
//
//      Coverage: 4 acts × 3 expert counts × 2 N1 widths = 24 runs.
//      For each: fused-MoE call with active = total/2 followed by
//      independent probes of Op1 (K=K_in, N=N1) and Op2 (K=K_op2,
//      N=H) caches.  Asserts cache_hits = total for both probes.
// ===============================================================================

struct KDownSynthParam {
  int act_int;             // 0=none, 1=silu_and_mul, 2=gelu_and_mul, 3=swiglu_oai_mul
  int total;               // total_matmul (full expert count)
  int N1;                  // Op1 N (gate+up width)
};

class TestPrepackKDownSynthesis :
    public ::testing::TestWithParam<KDownSynthParam> {};

TEST_P(TestPrepackKDownSynthesis, BothPassesWarmAllExperts) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Force ALGO=3 + custom-kernel ON so prepack_for_algo_3's body fires
  // its custom-kernel branch (the only side with observable HIT/MISS
  // counters).  Per-tile AOCL DLP warm also runs but is unobservable
  // through public API (tested separately in [18]).
  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  const auto &p = GetParam();
  const auto act = static_cast<grp_matmul_gated_act_t>(p.act_int);
  const int E         = p.total;
  const int K_active  = std::max(1, E / 2);
  const int N1        = p.N1;
  const int H         = 32;
  const int M         = 4;
  const int K_in      = H;
  const data_type_t dt = data_type_t::bf16;

  // Op2's K dimension follows the runtime contract in
  // group_matmul_fused_moe.cpp::op2_k_for_act():
  //   act == none → K_op2 = N1   (no compaction)
  //   any gated  → K_op2 = N1/2 (gate/up halves collapse to one)
  // The bug class is "Pass 2 prepack uses a different formula" — so
  // we drive every act through this single helper and assert hits.
  const int K_op2 = (act == grp_matmul_gated_act_t::none) ? N1 : (N1 / 2);

  // Per-expert weights with distinct buffers so each cache key is
  // unique (avoids the probe's warm-on-miss polluting the count).
  TypedBuffers w1_all, w2_all;
  w1_all.alloc(E, (size_t)K_in * N1, /*is_bf16=*/true);
  w2_all.alloc(E, (size_t)K_op2 * H, /*is_bf16=*/true);
  for (int e = 0; e < E; ++e) {
    fill_wei1(w1_all.bf16[e], e);
    fill_wei2(w2_all.bf16[e], e);
  }
  auto wei1_all = w1_all.cptrs(true);
  auto wei2_all = w2_all.cptrs(true);

  // Active prefix [0, K_active) gets meaningful src; tail experts
  // sit with M=0 (the framework contract for prepack-extras).
  TypedBuffers src_test;
  src_test.alloc(E, (size_t)M * K_in, true);
  for (int e = 0; e < K_active; ++e) fill_src(src_test.bf16[e], e);
  auto src_test_p = src_test.cptrs(true);

  auto gv = GemmVecs::uniform(E, M, N1, K_in);
  for (int e = K_active; e < E; ++e) gv.Ms[e] = 0;

  std::vector<const void *> no_bias(E, nullptr);
  auto params = make_uniform_params(E, dt);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(K_active);
    pp.total_matmul  = static_cast<uint32_t>(E);
  }

  grp_matmul_gated_act_params act_params{};
  act_params.act = act;
  auto act_ptr = (act != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  auto fused = make_fused_moe_op2(E, H, wei2_all, no_bias);

  std::vector<void *> dst_null(E, nullptr);
  std::vector<int>    ldc_null(E, 0);
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                src_test_p, gv.lda, wei1_all, gv.ldb,
                                no_bias, gv.beta, dst_null, ldc_null,
                                gv.is_wc, params, nullptr, act_ptr,
                                &fused),
            status_t::success);

  // ── Probe Op1: full N1 columns at K=K_in ──────────────────────
  std::vector<int>  K_op1_vec(E, K_in);
  std::vector<int>  N_op1_vec(E, N1);
  std::vector<int>  ldb_op1(E, N1);
  std::vector<bool> transB_op1(E, false);
  prepack::custom_kernel::PackProbeStats probe_op1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei1_all, K_op1_vec, N_op1_vec, ldb_op1, transB_op1,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op1);
  EXPECT_EQ(probe_op1.cache_hits, E)
      << "Op1 cache must be fully warmed (act=" << p.act_int
      << " E=" << E << " N1=" << N1 << ")";
  EXPECT_EQ(probe_op1.cache_misses, 0);

  // ── Probe Op2: K=op2_k_for_act(N1, act), N=H ──────────────────
  std::vector<int>  K_op2_vec(E, K_op2);
  std::vector<int>  N_op2_vec(E, H);
  std::vector<int>  ldb_op2(E, H);
  std::vector<bool> transB_op2(E, false);
  prepack::custom_kernel::PackProbeStats probe_op2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei2_all, K_op2_vec, N_op2_vec, ldb_op2, transB_op2,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op2);
  EXPECT_EQ(probe_op2.cache_hits, E)
      << "Op2 cache must be fully warmed for act=" << p.act_int
      << " E=" << E << " N1=" << N1 << " (K_op2=" << K_op2 << ")"
      << "; if cache_misses > 0, Pass 2 K_down synthesis is using "
         "a different formula than runtime op2_k_for_act() — this "
         "is the regression class of the original swiglu-only "
         "K_down bug, now possibly affecting "
      << (p.act_int == 0 ? "the act=none path"
        : p.act_int == 1 ? "silu_and_mul"
        : p.act_int == 2 ? "gelu_and_mul"
        :                  "swiglu_oai_mul");
  EXPECT_EQ(probe_op2.cache_misses, 0);
}

static std::vector<KDownSynthParam> make_k_down_synth_params() {
  std::vector<KDownSynthParam> out;
  // 4 acts × 3 expert counts × 2 N1 widths = 24 cases.
  // E=2 covers single-pair-of-experts edge; E=8 covers the typical
  // small-MoE; E=32 covers gpt-oss-20B's expert count.
  // N1 ∈ {64, 256} keeps per-case runtime sub-second while still
  // exercising both narrow (one tile) and wider (multi-tile) shapes.
  for (int act : {0, 1, 2, 3}) {
    for (int total : {2, 8, 32}) {
      for (int N1 : {64, 256}) {
        out.push_back({act, total, N1});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(
    GroupMatmulPrepackKDownSynth, TestPrepackKDownSynthesis,
    ::testing::ValuesIn(make_k_down_synth_params()),
    [](const ::testing::TestParamInfo<KDownSynthParam> &info) {
      const char *act = (info.param.act_int == 0) ? "none"
                      : (info.param.act_int == 1) ? "silu_and_mul"
                      : (info.param.act_int == 2) ? "gelu_and_mul"
                      :                             "swiglu_oai_mul";
      std::ostringstream s;
      s << "act_" << act << "_E" << info.param.total
        << "_N1_" << info.param.N1;
      return s.str();
    });

// ===============================================================================
// [21] TestPrepackResultInvariance - confidence-level regression net
//      that runs the SAME fused-MoE call twice (or under two different
//      scheduling ALGOs) and asserts the output stays equivalent.
//
//      The bug class targeted is "warm cache state corrupts runtime
//      computation" — e.g. weight buffer accidentally modified by a
//      prepack reorder, fingerprint cache returning stale state, or
//      per-tile cache key collision producing wrong weights at runtime.
//      Existing TestFusedMoE only validates ONE call vs the legacy
//      2-call reference; it doesn't catch a regression where the
//      first call is correct but a subsequent call with a warm cache
//      diverges.
//
//      Two TEST_P:
//        * ConsistentAcrossIterations - ALGO=3 + custom-kernel ON,
//          two consecutive calls.  First triggers prepack (cold
//          cache), second short-circuits via fingerprint cache.
//          Output must be bit-identical (matmul reduction order is
//          deterministic when ALGO + thread team are fixed and the
//          cache state is itself deterministic).
//
//        * Algo1VsAlgo3 - same inputs run with ALGO=1 then ALGO=3.
//          Prepack module fires for both (different scheduling_algo
//          fingerprints) so each path independently warms its
//          backend cache.  Output must match within BF16 fused-MoE
//          tolerance.  Catches algorithmic divergence (weight buffer
//          corrupted on one path but not the other, post-tile
//          accumulator reset wrong on per-tile path, etc.).
//
//      Coverage: 4 acts × 2 expert counts × 2 dim sizes = 16 cases
//      per TEST_P × 2 TEST_P = 32 runs total.  Sub-second per case;
//      total ~30s budget.
// ===============================================================================

struct InvarianceParam {
  int act_int;             // 0=none, 1=silu_and_mul, 2=gelu_and_mul, 3=swiglu_oai_mul
  int num_ops;             // total experts (also total_matmul)
  int dim;                 // half-width of gate+up: N_gate_up = 2 * dim
};

class TestPrepackResultInvariance :
    public ::testing::TestWithParam<InvarianceParam> {};

namespace {

// Shared MoE-call harness for the invariance tests.  Allocates one
// dst_down output buffer per call so the caller can compare across
// runs.  Returns the per-expert M vector (with M=0 for the inactive
// tail) so the caller's `verify_per_expert_2d(...)` can skip the
// non-written experts.
struct InvarianceHarness {
  moe_test_utils::TypedBuffers          src;
  moe_test_utils::TypedBuffers          w1;
  moe_test_utils::TypedBuffers          w2;
  std::vector<int>                      Ms;     // M=4 for active, M=0 tail
  int                                   N1     = 0;
  int                                   H      = 0;
  int                                   K_in   = 0;
  int                                   K_op2  = 0;
  int                                   active = 0;
  zendnnl::lowoha::matmul::grp_matmul_gated_act_t act =
      zendnnl::lowoha::matmul::grp_matmul_gated_act_t::none;
};

inline InvarianceHarness build_invariance_harness(
    const InvarianceParam &p) {
  using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
  InvarianceHarness h;
  h.act    = static_cast<grp_matmul_gated_act_t>(p.act_int);
  h.K_in   = 32;
  h.H      = 32;
  h.N1     = 2 * p.dim;
  h.K_op2  = (h.act == grp_matmul_gated_act_t::none) ? h.N1 : (h.N1 / 2);
  const int M = 4;
  const int E = p.num_ops;
  h.active = std::max(1, E / 2);

  h.src.alloc(E, (size_t)M * h.K_in,    /*is_bf16=*/true);
  h.w1 .alloc(E, (size_t)h.K_in * h.N1, true);
  h.w2 .alloc(E, (size_t)h.K_op2 * h.H, true);

  for (int e = 0; e < E; ++e) {
    moe_test_utils::fill_wei1(h.w1.bf16[e], e);
    moe_test_utils::fill_wei2(h.w2.bf16[e], e);
  }
  for (int e = 0; e < h.active; ++e) {
    moe_test_utils::fill_src(h.src.bf16[e], e);
  }

  h.Ms.assign(E, M);
  for (int e = h.active; e < E; ++e) h.Ms[e] = 0;
  return h;
}

// One fused-MoE invocation that captures dst_down into the caller's
// pre-allocated TypedBuffers.  Returns the dispatch status_t so the
// caller can ASSERT_EQ on success.
inline zendnnl::error_handling::status_t run_fused_moe_capture(
    const InvarianceHarness &h, int num_ops,
    moe_test_utils::TypedBuffers &dst_out) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::error_handling::status_t;

  auto srcs = h.src.cptrs(true);
  auto wei1 = h.w1.cptrs(true);
  auto wei2 = h.w2.cptrs(true);
  auto dst_p = dst_out.ptrs(true);

  auto gv = GemmVecs::uniform(num_ops, /*M=*/4, h.N1, h.K_in);
  gv.Ms = h.Ms;

  std::vector<const void *> no_bias(num_ops, nullptr);
  auto params = make_uniform_params(num_ops,
                                    zendnnl::common::data_type_t::bf16);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(h.active);
    pp.total_matmul  = static_cast<uint32_t>(num_ops);
  }

  grp_matmul_gated_act_params act_params{};
  act_params.act = h.act;
  auto act_ptr =
      (h.act != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  auto fused = make_fused_moe_op2(num_ops, h.H, wei2, no_bias);
  fused.dst_down = dst_p;
  fused.ldc_down = std::vector<int>(num_ops, h.H);

  std::vector<void *> dst_op1_null(num_ops, nullptr);
  std::vector<int>    ldc_null(num_ops, 0);

  return group_matmul_direct(
      gv.layout, gv.transA, gv.transB,
      gv.Ms, gv.Ns, gv.Ks, gv.alpha,
      srcs, gv.lda, wei1, gv.ldb,
      no_bias, gv.beta, dst_op1_null, ldc_null,
      gv.is_wc, params, nullptr, act_ptr, &fused);
}

} // namespace

TEST_P(TestPrepackResultInvariance, ConsistentAcrossIterations) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // Pin to ALGO=3 + custom-kernel ON so all four prepack code paths
  // engage on the first call (custom-kernel BF16 pack + AOCL DLP
  // per-tile when STABLE_NTILE is on).  The second call must
  // short-circuit via fingerprint cache and produce identical output.
  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  const auto &p = GetParam();
  auto h = build_invariance_harness(p);

  TypedBuffers dst_a, dst_b;
  dst_a.alloc(p.num_ops, (size_t)4 * h.H, /*is_bf16=*/true);
  dst_b.alloc(p.num_ops, (size_t)4 * h.H, /*is_bf16=*/true);

  // Call 1: cold cache → prepack fires, runtime runs.
  ASSERT_EQ(run_fused_moe_capture(h, p.num_ops, dst_a),
            status_t::success);

  // Call 2: warm cache + fingerprint cache hit → prepack short-
  // circuits, runtime runs from cached weights.  Same inputs, same
  // ALGO, deterministic reduction order ⇒ bit-identical output.
  ASSERT_EQ(run_fused_moe_capture(h, p.num_ops, dst_b),
            status_t::success);

  std::ostringstream lbl;
  lbl << "act=" << p.act_int << " E=" << p.num_ops
      << " dim=" << p.dim << " (cold→warm consistency)";
  // Tol{0,0} = bit-identical.  If this fails, prepack is mutating
  // state that runtime depends on (weight buffer corruption,
  // fingerprint cache returning stale data, per-tile key collision
  // returning wrong reordered weights, etc.).
  verify_per_expert_2d(dst_a, h.H, dst_b, h.H,
                       h.Ms, h.H, /*is_bf16=*/true,
                       Tol{0.0f, 0.0f}, lbl.str());
}

TEST_P(TestPrepackResultInvariance, Algo1VsAlgo3) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  const auto &p = GetParam();
  auto h = build_invariance_harness(p);

  TypedBuffers dst_a1, dst_a3;
  dst_a1.alloc(p.num_ops, (size_t)4 * h.H, /*is_bf16=*/true);
  dst_a3.alloc(p.num_ops, (size_t)4 * h.H, /*is_bf16=*/true);

  // ALGO 1 (sequential_experts): full-weight reorder cache, no
  // tile decomposition.
  {
    AlgoEnvGuard algo_guard(1);
    ASSERT_EQ(run_fused_moe_capture(h, p.num_ops, dst_a1),
              status_t::success);
  }

  // ALGO 3 (flat_n_tile): per-tile reorder cache under strict-stable
  // plan, custom-kernel BF16 fast path when env says so.  Different
  // scheduling_algo fingerprint → independent prepack pass.
  reset_grp_matmul_caches();
  {
    AlgoEnvGuard algo_guard(3);
    ASSERT_EQ(run_fused_moe_capture(h, p.num_ops, dst_a3),
              status_t::success);
  }

  std::ostringstream lbl;
  lbl << "act=" << p.act_int << " E=" << p.num_ops
      << " dim=" << p.dim << " (ALGO 1 vs ALGO 3 result match)";
  // Use the standard fused-MoE BF16 tolerance — algorithmic
  // divergence between the per-expert sequential path and the
  // per-tile parallel path is allowed within the same envelope as
  // any cross-algo comparison in this gtest.
  verify_per_expert_2d(dst_a1, h.H, dst_a3, h.H,
                       h.Ms, h.H, /*is_bf16=*/true,
                       tol_fused(/*is_bf16=*/true), lbl.str());
}

static std::vector<InvarianceParam> make_invariance_params() {
  std::vector<InvarianceParam> out;
  // 4 acts × 2 expert counts × 2 dim sizes = 16 cases per TEST_P.
  // E ∈ {4, 16} covers both small-MoE and decode-class.
  // dim ∈ {32, 64} keeps per-case runtime sub-second while still
  // exercising both narrow-N (single tile) and wider (multi-tile)
  // ALGO 3 strict-stable plans.
  for (int act : {0, 1, 2, 3}) {
    for (int E : {4, 16}) {
      for (int dim : {32, 64}) {
        out.push_back({act, E, dim});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(
    GroupMatmulPrepackResultInvariance, TestPrepackResultInvariance,
    ::testing::ValuesIn(make_invariance_params()),
    [](const ::testing::TestParamInfo<InvarianceParam> &info) {
      const char *act = (info.param.act_int == 0) ? "none"
                      : (info.param.act_int == 1) ? "silu_and_mul"
                      : (info.param.act_int == 2) ? "gelu_and_mul"
                      :                             "swiglu_oai_mul";
      std::ostringstream s;
      s << "act_" << act << "_E" << info.param.num_ops
        << "_dim_" << info.param.dim;
      return s.str();
    });

// ===============================================================================
// [22] TestPrepackClearCacheDirect - direct regression tests for the
//      cache-clear APIs (`clear_custom_kernel_pack_cache()` and
//      `clear_fingerprint_cache_for_test()`).
//
//      Targeted bug class: the `clear_custom_kernel_pack_cache()` size_t
//      underflow regression that lets the LRU fast-path on
//      `set_capacity(0)` → `set_capacity(MAX)` without iterating any
//      eviction.  Symptom: tests that re-use heap addresses across
//      runs see stale cache entries and report false HITS.  Fix
//      replaced the pair with `pack_cache.clear()`.
//
//      Two TEST_F:
//        * CustomKernelCacheClearEvictsEntries — populate cache,
//          clear, re-probe → all MISSES.  If the underflow returns,
//          re-probe sees HITS.
//        * FingerprintClearEnablesPrepackReFire — exercise the
//          documented contract that `clear_fingerprint_cache_for_test()`
//          unblocks a re-warm even when the underlying caches were
//          already populated.  Verifies the fingerprint cache is the
//          load-bearing short-circuit (not "warm-pack runs always").
// ===============================================================================

class TestPrepackClearCacheDirect : public ::testing::Test {};

TEST_F(TestPrepackClearCacheDirect, CustomKernelCacheClearEvictsEntries) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.4f);

  // Cold cache → first probe records 4 misses + warms entries.
  prepack::custom_kernel::PackProbeStats st_cold;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_cold);
  EXPECT_EQ(st_cold.cache_misses, 4);
  EXPECT_EQ(st_cold.packed_ok,    4);

  // Warm cache → second probe is all hits.
  prepack::custom_kernel::PackProbeStats st_warm;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_warm);
  EXPECT_EQ(st_warm.cache_hits,   4);
  EXPECT_EQ(st_warm.cache_misses, 0);

  // Clear: every entry must be evicted.  If `clear()` regresses to
  // `set_capacity(0)` followed by `set_capacity(MAX)` the LRU's
  // `evict(0 - MAX)` underflows (size_t) and never iterates, so the
  // post-clear probe sees HITS instead of MISSES.
  custom_kernel::clear_custom_kernel_pack_cache();

  prepack::custom_kernel::PackProbeStats st_after_clear;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_after_clear);
  EXPECT_EQ(st_after_clear.cache_misses, 4)
      << "clear_custom_kernel_pack_cache() must evict all entries; "
         "if cache_misses < 4 the size_t-underflow regression is back "
         "(set_capacity(0) → set_capacity(MAX) which never iterates evict).";
  EXPECT_EQ(st_after_clear.cache_hits, 0);
}

TEST_F(TestPrepackClearCacheDirect, FingerprintClearEnablesPrepackReFire) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/32, /*N=*/64, /*fill=*/0.55f);

  // Call 1: prepack fires (cold fingerprint cache).  ALGO 3 + the
  // BF16/BF16/BF16 dtype triple drive the custom-kernel branch
  // (custom_kernel_on=true is set by `make_harness` on the harness
  // PrepackParams).
  prepack::prepack_for_algo_3(h.pp);

  // Probe: all 4 should be HITS (warmed by call 1).
  prepack::custom_kernel::PackProbeStats st_after_call1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_after_call1);
  EXPECT_EQ(st_after_call1.cache_hits, 4)
      << "first prepack_for_algo_3 should have populated all entries";

  // Wipe the custom-kernel cache only (fingerprint cache stays).
  // The cache is now COLD again but the per-thread fingerprint set
  // still says "we already warmed this configuration".
  custom_kernel::clear_custom_kernel_pack_cache();

  // Call 2: prepack short-circuits via fingerprint cache, leaves
  // the (cleared) custom-kernel cache cold.  Verifiable via the
  // probe immediately after.
  prepack::prepack_for_algo_3(h.pp);

  prepack::custom_kernel::PackProbeStats st_after_call2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_after_call2);
  EXPECT_EQ(st_after_call2.cache_misses, 4)
      << "second prepack_for_algo_3 with same fingerprint should "
         "short-circuit; if it had re-warmed, cache_misses would be 0 "
         "and cache_hits would be 4.";

  // Now clear BOTH caches: custom-kernel + fingerprint.
  reset_grp_matmul_caches();

  // Call 3: fingerprint MISS (was cleared) → prepack fires → custom-
  // kernel cache populates again.
  prepack::prepack_for_algo_3(h.pp);

  prepack::custom_kernel::PackProbeStats st_after_call3;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, st_after_call3);
  EXPECT_EQ(st_after_call3.cache_hits, 4)
      << "after clear_fingerprint_cache_for_test(), prepack must re-fire "
         "and re-populate the custom-kernel cache.  If misses>0, the "
         "fingerprint cache isn't being cleared — every gtest case that "
         "calls clear_fingerprint_cache_for_test() at start would fail "
         "to isolate from prior tests' fingerprint state.";
}

// ===============================================================================
// [23] TestPrepackAoclDlpFullWeightConstGate - mirrors the existing
//      per-tile `AoclDlpNTileSkipsNonConstExperts` test for the
//      FULL-WEIGHT warmer path (`warm_pack_all_aocl_dlp_experts`).
//
//      Targeted bug class B5: the AOCL DLP warmer used to populate
//      its LRU for every expert regardless of `is_weights_const[e]`,
//      while the runtime `run_dlp(...)` only consults the cache when
//      `is_weights_const` is true (aocl_kernel.cpp:1700-1702).  The
//      fix added a parity gate at the warmer's per-expert loop;
//      regression here would re-introduce wasted cache entries the
//      runtime never queries (memory bloat + extra reorder cost).
//
//      The per-tile path now has its own dedicated test; this section
//      closes the gap on the full-weight path used by ALGOs 1, 2, 4,
//      5 and by ALGO 3 fallbacks (STABLE_NTILE off, narrow-N escape).
// ===============================================================================

class TestPrepackAoclDlpFullWeight : public ::testing::Test {};

TEST_F(TestPrepackAoclDlpFullWeight, SkipsNonConstExperts) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.875f);
  // Mark experts 1 + 3 as variable-weight — `run_dlp(...)` won't
  // cache those at runtime (aocl_kernel.cpp:1700-1702), so the
  // warmer must skip them too.
  h.is_weights_const[1] = false;
  h.is_weights_const[3] = false;

  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts(
          h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
          /*total_count=*/4,
          /*wei_dtype=*/data_type_t::bf16,
          st),
      status_t::success);

  // Two valid const experts (0, 2) → packed_ok = 2.
  // Two non-const (1, 3) → skipped_invalid = 2.
  // total_attempted counts every iteration regardless of skip path.
  EXPECT_EQ(st.total_attempted, 4);
  EXPECT_EQ(st.packed_ok,        2)
      << "full-weight warmer must skip variable-weight experts to "
         "stay in parity with `run_dlp(...)`'s `is_weights_const` "
         "gate (aocl_kernel.cpp:1700-1702).  Regressing this re-"
         "introduces wasted cache entries the dispatcher never "
         "consults, bloating LRU + duplicating reorder cost.";
  EXPECT_EQ(st.skipped_invalid,  2);
}

TEST_F(TestPrepackAoclDlpFullWeight, EmptyIsConstTreatsAllAsConst) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  auto h = make_harness(/*total=*/4, /*active=*/2,
                        /*K=*/64, /*N=*/128, /*fill=*/0.375f);

  // Empty `is_weights_const` is the documented "treat every entry
  // as const" sentinel for legacy callers that don't pass the field.
  // The warmer's contract guarantees it processes all entries.
  std::vector<bool> empty_iwc;
  prepack::aocl_dlp::AoclDlpPackProbeStats st;
  ASSERT_EQ(
      prepack::aocl_dlp::warm_pack_all_aocl_dlp_experts(
          h.weight, h.K, h.N, h.ldb, h.transB, empty_iwc,
          /*total_count=*/4,
          /*wei_dtype=*/data_type_t::bf16,
          st),
      status_t::success);

  EXPECT_EQ(st.total_attempted,  4);
  EXPECT_EQ(st.packed_ok,        4);
  EXPECT_EQ(st.skipped_invalid,  0);
}

// ===============================================================================
// [24] TestPrepackVariableNExperts - per-expert N skew test.
//
//      Targeted bug class B11: a future refactor that uniformises
//      `scratch.K_down` (e.g. uses `op2_k_for_act(N[0], act)` for
//      every expert instead of `op2_k_for_act(N[e], act)` per
//      expert) would cache-miss on every entry beyond e=0 when the
//      framework feeds an MoE block with non-uniform expert widths.
//      No production model uses variable-N today (every expert in a
//      block shares the same gate+up width), but treating that as a
//      hard contract instead of a defensive invariant has bitten us
//      before.  This test pins it down.
//
//      Setup: E=4, alternating N1 ∈ {64, 128, 64, 128}.  Op1 cache
//      keys MUST be at `(K=K_in, N=N[e])` per expert; Op2 keys at
//      `(K=op2_k_for_act(N[e], act), N=H)`.  Both probes assert
//      cache_hits = E.
// ===============================================================================

class TestPrepackVariableN : public ::testing::Test {};

TEST_F(TestPrepackVariableN, MixedNAcrossExperts) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::common::data_type_t;

  reset_grp_matmul_caches();

  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  const int E         = 4;
  const int K_active  = 2;        // active prefix (gv.Ms[0..1] = M, [2..3] = 0)
  const int M         = 4;
  const int K_in      = 32;
  const int H         = 32;
  const data_type_t dt = data_type_t::bf16;
  const auto act      = grp_matmul_gated_act_t::silu_and_mul;

  // Alternating N → exposes any uniformised K_down derivation.
  std::vector<int> N_op1 = {64, 128, 64, 128};
  std::vector<int> K_op2(E);
  for (int e = 0; e < E; ++e) {
    // op2_k_for_act(N, silu_and_mul) = N/2 per expert.
    K_op2[e] = N_op1[e] / 2;
  }

  // Per-expert weight buffers sized to each expert's own (K, N).
  std::vector<std::vector<bfloat16_t>> w1_storage(E), w2_storage(E);
  std::vector<const void *> wei1_all(E), wei2_all(E);
  for (int e = 0; e < E; ++e) {
    w1_storage[e].assign(static_cast<size_t>(K_in) * N_op1[e],
                         bfloat16_t(0.005f + 0.001f * e));
    w2_storage[e].assign(static_cast<size_t>(K_op2[e]) * H,
                         bfloat16_t(0.008f + 0.001f * e));
    wei1_all[e] = w1_storage[e].data();
    wei2_all[e] = w2_storage[e].data();
  }

  TypedBuffers src_test;
  src_test.alloc(E, (size_t)M * K_in, /*is_bf16=*/true);
  for (int e = 0; e < K_active; ++e) fill_src(src_test.bf16[e], e);
  auto src_test_p = src_test.cptrs(true);

  // GemmVecs with per-expert Ns and ldb (=N for non-transposed B).
  // `is_wc = true` so the CK pack cache + AOCL DLP cache are
  // populated (this test's CK probes below assert hits at per-
  // expert N keys; under `is_wc = false` the runtime CK refusal
  // gate at `custom_kernel/dispatch.cpp::prepare_for_call` would
  // skip CK engagement entirely and the cache would stay empty).
  GemmVecs gv;
  gv.layout.assign(E, 'r');
  gv.transA.assign(E, false);
  gv.transB.assign(E, false);
  gv.is_wc .assign(E, true);
  gv.alpha .assign(E, 1.0f);
  gv.beta  .assign(E, 0.0f);
  gv.Ms.assign(E, M);
  for (int e = K_active; e < E; ++e) gv.Ms[e] = 0;
  gv.Ns  = N_op1;
  gv.Ks .assign(E, K_in);
  gv.lda.assign(E, K_in);
  gv.ldb = N_op1;
  gv.ldc = N_op1;

  std::vector<const void *> no_bias(E, nullptr);
  auto params = make_uniform_params(E, dt);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(K_active);
    pp.total_matmul  = static_cast<uint32_t>(E);
  }

  grp_matmul_gated_act_params act_params{};
  act_params.act = act;

  // Per-expert N_down and ldb_down stay uniform (H is fixed across
  // experts in a real MoE block — every expert outputs the same
  // hidden dimension).  Per-expert K_op2 is the only width that
  // varies post-activation.
  auto fused = make_fused_moe_op2(E, H, wei2_all, no_bias);

  std::vector<void *> dst_null(E, nullptr);
  std::vector<int>    ldc_null(E, 0);
  ASSERT_EQ(group_matmul_direct(gv.layout, gv.transA, gv.transB,
                                gv.Ms, gv.Ns, gv.Ks, gv.alpha,
                                src_test_p, gv.lda, wei1_all, gv.ldb,
                                no_bias, gv.beta, dst_null, ldc_null,
                                gv.is_wc, params, nullptr, &act_params,
                                &fused),
            status_t::success);

  // Op1 probe: per-expert N + ldb must match the cache key the
  // dispatcher built at runtime.
  std::vector<int>  K_op1_vec(E, K_in);
  std::vector<bool> transB_vec(E, false);
  prepack::custom_kernel::PackProbeStats probe_op1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei1_all, K_op1_vec, N_op1, /*ldb=*/N_op1, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op1);
  EXPECT_EQ(probe_op1.cache_hits, E)
      << "Op1 cache must hit at per-expert (K=" << K_in << ", N[e]) "
         "key.  cache_misses>0 means Pass 1 prepack uniformised the "
         "Op1 widths instead of forwarding the per-expert N vector.";

  // Op2 probe: per-expert K_op2 (= N[e]/2 for silu_and_mul) drives
  // the cache key.  This is the regression class — a uniformised
  // K_down (e.g. always N[0]/2 = 32) would HIT only the experts
  // whose actual K_op2 happened to be 32, MISS the rest.
  std::vector<int>  N_op2_vec(E, H);
  std::vector<int>  ldb_op2_vec(E, H);
  prepack::custom_kernel::PackProbeStats probe_op2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      wei2_all, K_op2, N_op2_vec, ldb_op2_vec, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op2);
  EXPECT_EQ(probe_op2.cache_hits, E)
      << "Op2 cache must hit at per-expert K_op2 = op2_k_for_act(N[e], act). "
         "cache_misses>0 indicates scratch.K_down has been uniformised "
         "(e.g. uses N[0]/2 for every expert) instead of being computed "
         "per-expert; downstream the runtime would build cache keys at "
         "the actual N[e]/2 and miss the cached entries Pass 2 populated "
         "under the wrong K.";
}

// ===============================================================================
// [25] TestPrepackStressManyExperts - confidence at scale.
//
//      Two scenarios:
//        * E=64 multi-iteration: 5 fused-MoE calls back-to-back.
//          First iteration warms the cache; iterations 2-5 short-
//          circuit via fingerprint cache.  Final probe asserts all
//          64 entries are HITS — i.e., the LRU never evicted entries
//          under steady-state pressure.  Catches any silent eviction
//          regression (e.g., capacity off-by-one) that would surface
//          at a model with 32-64 experts after a few decode tokens.
//
//        * E=256 single-iteration: stress the kNTilePlanMaxExperts
//          boundary.  256 is the planner's hardcoded max and the
//          stable_n_thr_per_expert array's upper bound (defined as
//          kNTilePlanMaxExperts in group_matmul_parallel_common.hpp,
//          aliased on GroupNTilePlan::kMaxExperts).
//          A regression that off-by-ones the boundary check would
//          truncate prepack to 255 entries; the post-call probe
//          would see 1 MISS at e=255.
// ===============================================================================

class TestPrepackStress : public ::testing::Test {};

namespace {

// Run a single fused-MoE call with `total` total experts, `active`
// firing prefix, uniform shapes.  Caller-allocated weight banks
// shared via the harness.  Returns the dispatch status_t.
struct StressHarness {
  std::vector<std::vector<bfloat16_t>> w1_banks;
  std::vector<std::vector<bfloat16_t>> w2_banks;
  std::vector<const void *>            wei1;
  std::vector<const void *>            wei2;
  moe_test_utils::TypedBuffers         src;
  std::vector<int>                     Ms;
  int                                  E       = 0;
  int                                  active  = 0;
  int                                  M       = 0;
  int                                  K_in    = 0;
  int                                  N1      = 0;
  int                                  H       = 0;
};

inline StressHarness build_stress_harness(int E, int active) {
  StressHarness h;
  h.E      = E;
  h.active = active;
  h.M      = 2;        // small M to keep matmul cost minimal at E=256
  h.K_in   = 32;
  h.N1     = 64;
  h.H      = 32;
  const int K_op2 = h.N1 / 2;   // silu_and_mul

  h.w1_banks.resize(E);
  h.w2_banks.resize(E);
  h.wei1.resize(E);
  h.wei2.resize(E);
  for (int e = 0; e < E; ++e) {
    h.w1_banks[e].assign((size_t)h.K_in * h.N1,
                         bfloat16_t(0.005f + 1e-5f * (float)e));
    h.w2_banks[e].assign((size_t)K_op2 * h.H,
                         bfloat16_t(0.008f + 1e-5f * (float)e));
    h.wei1[e] = h.w1_banks[e].data();
    h.wei2[e] = h.w2_banks[e].data();
  }

  h.src.alloc(E, (size_t)h.M * h.K_in, /*is_bf16=*/true);
  for (int e = 0; e < active; ++e) {
    moe_test_utils::fill_src(h.src.bf16[e], e);
  }

  h.Ms.assign(E, h.M);
  for (int e = active; e < E; ++e) h.Ms[e] = 0;
  return h;
}

inline zendnnl::error_handling::status_t run_stress_call(
    const StressHarness &h) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  auto srcs = h.src.cptrs(true);
  auto gv = GemmVecs::uniform(h.E, h.M, h.N1, h.K_in);
  gv.Ms = h.Ms;

  std::vector<const void *> no_bias(h.E, nullptr);
  auto params = make_uniform_params(h.E, data_type_t::bf16);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(h.active);
    pp.total_matmul  = static_cast<uint32_t>(h.E);
  }

  grp_matmul_gated_act_params act_params{};
  act_params.act = grp_matmul_gated_act_t::silu_and_mul;

  auto fused = make_fused_moe_op2(h.E, h.H, h.wei2, no_bias);

  std::vector<void *> dst_null(h.E, nullptr);
  std::vector<int>    ldc_null(h.E, 0);
  return group_matmul_direct(
      gv.layout, gv.transA, gv.transB,
      gv.Ms, gv.Ns, gv.Ks, gv.alpha,
      srcs, gv.lda, h.wei1, gv.ldb,
      no_bias, gv.beta, dst_null, ldc_null,
      gv.is_wc, params, nullptr, &act_params, &fused);
}

} // namespace

TEST_F(TestPrepackStress, E64MultiIterationCacheStable) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  // 64 experts, 8 firing each iteration.  Realistic decode-class
  // setting for gpt-oss-20B (32 experts, top-4) or DeepSeek-V2
  // (160 experts, top-6 selected) sized down for quick test runtime.
  constexpr int E      = 64;
  constexpr int active = 8;
  auto h = build_stress_harness(E, active);

  // Five iterations: first warms the cache, the rest must be no-ops
  // under fingerprint short-circuit.  If LRU silently evicts an
  // entry between iterations, the post-loop probe will catch it as
  // a MISS.
  for (int it = 0; it < 5; ++it) {
    ASSERT_EQ(run_stress_call(h), status_t::success)
        << "fused-MoE call failed at iteration " << it;
  }

  // Probe Op1 (full E entries, all should be cached).
  const int K_op2 = h.N1 / 2;
  std::vector<int>  K_op1_vec(E, h.K_in);
  std::vector<int>  N_op1_vec(E, h.N1);
  std::vector<int>  ldb_op1_vec(E, h.N1);
  std::vector<bool> transB_vec(E, false);
  prepack::custom_kernel::PackProbeStats probe_op1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.wei1, K_op1_vec, N_op1_vec, ldb_op1_vec, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op1);
  EXPECT_EQ(probe_op1.cache_hits, E)
      << "all 64 Op1 entries must remain cached after 5 iterations; "
         "cache_misses>0 implies the LRU evicted entries (capacity "
         "regression) or fingerprint cache failed to short-circuit.";

  // Probe Op2 (full E entries).
  std::vector<int>  K_op2_vec(E, K_op2);
  std::vector<int>  N_op2_vec(E, h.H);
  std::vector<int>  ldb_op2_vec(E, h.H);
  prepack::custom_kernel::PackProbeStats probe_op2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.wei2, K_op2_vec, N_op2_vec, ldb_op2_vec, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op2);
  EXPECT_EQ(probe_op2.cache_hits, E)
      << "all 64 Op2 entries must remain cached after 5 iterations";
}

TEST_F(TestPrepackStress, E256BoundaryAllExpertsWarmed) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  AlgoEnvGuard algo_guard(3);
  EnvVarGuard custom_guard("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1");

  // 256 experts is `kNTilePlanMaxExperts` — the planner's hardcoded
  // upper bound for `stable_n_thr_per_expert` (single source of
  // truth in `group_matmul_parallel_common.hpp`, aliased on
  // `GroupNTilePlan::kMaxExperts`).  A regression that off-by-ones
  // the boundary check would truncate prepack to 255 entries; the
  // probe at e=255 catches that.
  constexpr int E      = 256;
  constexpr int active = 8;
  auto h = build_stress_harness(E, active);

  ASSERT_EQ(run_stress_call(h), status_t::success);

  const int K_op2 = h.N1 / 2;
  std::vector<int>  K_op1_vec(E, h.K_in);
  std::vector<int>  N_op1_vec(E, h.N1);
  std::vector<int>  ldb_op1_vec(E, h.N1);
  std::vector<bool> transB_vec(E, false);
  prepack::custom_kernel::PackProbeStats probe_op1;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.wei1, K_op1_vec, N_op1_vec, ldb_op1_vec, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op1);
  EXPECT_EQ(probe_op1.cache_hits, E)
      << "all 256 (kNTilePlanMaxExperts) Op1 entries must be warmed "
         "post-call; an off-by-one on the upper bound truncates "
         "prepack to 255 and leaves entry 255 cold.";

  std::vector<int>  K_op2_vec(E, K_op2);
  std::vector<int>  N_op2_vec(E, h.H);
  std::vector<int>  ldb_op2_vec(E, h.H);
  prepack::custom_kernel::PackProbeStats probe_op2;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.wei2, K_op2_vec, N_op2_vec, ldb_op2_vec, transB_vec,
      /*is_weights_const=*/std::vector<bool>{},
      /*total_count=*/E, probe_op2);
  EXPECT_EQ(probe_op2.cache_hits, E)
      << "all 256 Op2 entries must be warmed at the boundary";
}

// ===============================================================================
// [26] [27] [28] Env-knob matrix coverage (subprocess-isolated).
//
//      Most `ZENDNNL_GRP_MATMUL_*` env knobs are read once via
//      `static const v = []() { getenv(...); }()` at first call and
//      cached for the process lifetime.  In-process `setenv` after
//      that first read is a no-op.  To exercise every value combination
//      we use gtest's `EXPECT_EXIT` with `threadsafe` death-test style:
//      gtest `fork()`s and `execve()`s the binary, the child runs from
//      a fresh image (every `static const` reinitialised), and the
//      child inherits the parent's process environment via execve so
//      our `setenv` in the parent is visible to the child's first
//      env-getter read.
//
//      Three sections, sharing one harness:
//        [26] TestPrepackEnvBucketA  - perf-critical knobs
//             (PREPACK, STABLE_NTILE, CUSTOM_KERNEL, ALGO,
//              WEIGHT_CACHE, FUSED_MOE_TIGHT, N_TILE_FUSED_ACT,
//              N_ROUNDS, CUSTOM_KERNEL_NR).  Each value tested for
//              numerical correctness against the 2-call reference.
//        [27] TestPrepackEnvBucketB  - tuning knobs
//             (AOCL_TARGET_SLOTS, AOCL_BLIS_NC, N_ORDER,
//              CUSTOM_KERNEL_SUBTILE_PER_EXPERT,
//              CUSTOM_KERNEL_N_TILE).  Smoke + correctness — these
//              don't change production wall-time but may interact
//              with the prepack module's per-tile sizing.
//        [28] TestPrepackEnvInteractionMatrix - cross-product corners
//             of the most orthogonal Bucket-A knobs (12 cases) so
//             interaction bugs the single-knob sweep misses get
//             surfaced.  Production fast path, AOCL-only,
//             STABLE_NTILE-off, WEIGHT_CACHE-off, kill-switch, etc.
//
//      Subprocess overhead is ~150ms per case.  Total ~50 cases →
//      ~7-8s of additional wall time on the suite.
// ===============================================================================

namespace {

// One env knob preset.  `value` is forwarded directly to setenv so use
// "0" / "1" / "2" / etc. as appropriate.  Use {} (empty string) to
// force-unset (the helper turns it into unsetenv).
struct EnvCase {
  std::vector<std::pair<const char *, const char *>> envs;
  std::string                                        label;
  bool                                               check_correctness = true;
};

inline EnvCase env_case(
    std::string label,
    std::vector<std::pair<const char *, const char *>> envs,
    bool check = true) {
  EnvCase c;
  c.label = std::move(label);
  c.envs  = std::move(envs);
  c.check_correctness = check;
  return c;
}

inline void apply_env_preset(const EnvCase &p) {
  for (const auto &kv : p.envs) {
    setenv(kv.first, kv.second, /*overwrite=*/1);
  }
}
inline void unapply_env_preset(const EnvCase &p) {
  for (const auto &kv : p.envs) {
    unsetenv(kv.first);
  }
}

// Tiny fused-MoE shape used by every env-matrix subprocess case.
// Small enough that the matmul work is dwarfed by the fork+exec
// overhead; large enough to exercise per-tile decomposition and
// gated activation in the prepack module.
struct EnvMatrixShape {
  int E         = 4;
  int active    = 2;
  int M         = 4;
  int K_in      = 32;
  int N_gate_up = 64;
  int H         = 32;
  zendnnl::lowoha::matmul::grp_matmul_gated_act_t act =
      zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;
};

// Child-side body: run fused-MoE + 2-call reference under the
// inherited env, compare per-expert dst element-wise.  gtest
// assertions inside record into the child process's gtest test-state;
// the caller wraps with `std::exit(::testing::Test::HasFailure() ? 1 : 0)`
// so the parent's `EXPECT_EXIT` sees a non-zero exit on any failure.
inline void run_fused_vs_ref_in_child(
    const EnvCase &p,
    const EnvMatrixShape &shape) {
  using namespace zendnnl::lowoha::matmul;
  using namespace moe_test_utils;
  using zendnnl::common::data_type_t;

  const int  E         = shape.E;
  const int  K_active  = shape.active;
  const int  M         = shape.M;
  const int  K_in      = shape.K_in;
  const int  N1        = shape.N_gate_up;
  const int  H         = shape.H;
  const auto act       = shape.act;
  const int  K_op2     = (act == grp_matmul_gated_act_t::none) ? N1 : N1 / 2;
  const data_type_t dt = data_type_t::bf16;

  TypedBuffers src, w1, d1_ref, w2, d2_fused, d2_ref;
  src     .alloc(E, (size_t)M * K_in,    /*is_bf16=*/true);
  w1      .alloc(E, (size_t)K_in * N1,    true);
  d1_ref  .alloc(E, (size_t)M * N1,       true);
  w2      .alloc(E, (size_t)K_op2 * H,    true);
  d2_fused.alloc(E, (size_t)M * H,        true);
  d2_ref  .alloc(E, (size_t)M * H,        true);

  for (int e = 0; e < E; ++e) {
    fill_wei1(w1.bf16[e], e);
    fill_wei2(w2.bf16[e], e);
  }
  for (int e = 0; e < K_active; ++e) {
    fill_src(src.bf16[e], e);
  }

  std::vector<int> Ms(E, M);
  for (int e = K_active; e < E; ++e) Ms[e] = 0;

  auto srcs       = src.cptrs(true);
  auto wei1       = w1.cptrs(true);
  auto wei2       = w2.cptrs(true);
  auto dst1_ref_p = d1_ref.ptrs(true);
  auto dst2_f     = d2_fused.ptrs(true);
  auto dst2_r     = d2_ref.ptrs(true);
  std::vector<const void *> no_bias(E, nullptr);

  // ── Reference: 2-call decomposition ─────────────────────────────
  // Same env state as fused (we're already in the child process).
  // Any env-induced numerical drift applies symmetrically to both
  // sides of the comparison, so a cross-env compare stays meaningful.
  ASSERT_EQ(run_legacy_2call_ref(Ms, K_in, N1, K_op2, H,
                                 /*is_bf16=*/true, act,
                                 srcs, wei1, wei2, dst1_ref_p, dst2_r),
            status_t::success)
      << "[env=" << p.label << "] reference 2-call dispatch failed";

  // ── Fused dispatcher under prepack module ───────────────────────
  auto params = make_uniform_params(E, dt);
  for (auto &pp : params) {
    pp.active_matmul = static_cast<uint32_t>(K_active);
    pp.total_matmul  = static_cast<uint32_t>(E);
  }

  auto gv_op1 = GemmVecs::uniform(E, M, N1, K_in);
  gv_op1.Ms = Ms;

  grp_matmul_gated_act_params act_params{};
  act_params.act = act;
  auto act_ptr = (act != grp_matmul_gated_act_t::none) ? &act_params : nullptr;

  auto fused = make_fused_moe_op2(E, H, wei2, no_bias);
  fused.dst_down = dst2_f;
  fused.ldc_down = std::vector<int>(E, H);

  auto pf = params;
  std::vector<void *> dst_op1_null(E, nullptr);
  std::vector<int>    ldc_null(E, 0);
  ASSERT_EQ(group_matmul_direct(gv_op1.layout, gv_op1.transA, gv_op1.transB,
                                gv_op1.Ms, gv_op1.Ns, gv_op1.Ks, gv_op1.alpha,
                                srcs, gv_op1.lda, wei1, gv_op1.ldb,
                                no_bias, gv_op1.beta, dst_op1_null, ldc_null,
                                gv_op1.is_wc, pf, nullptr, act_ptr, &fused),
            status_t::success)
      << "[env=" << p.label << "] fused dispatch failed";

  // ── Compare ───────────────────────────────────────────────────────
  if (p.check_correctness) {
    std::ostringstream lbl;
    lbl << "[env=" << p.label << "]";
    verify_per_expert_2d(d2_fused, H, d2_ref, H,
                         Ms, H, /*is_bf16=*/true,
                         tol_fused(/*is_bf16=*/true), lbl.str());
  }
}

// Per-test driver shared by Bucket A / B / Interaction matrix.
inline void run_env_matrix_subprocess_test(const EnvCase &p) {
  // Force the subprocess fork+exec path so the child gets a fresh
  // process image (every `static const` getter reinitialised).  The
  // default "fast" mode uses fork() only, which clones the parent's
  // already-cached `static const` and defeats the purpose.
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  apply_env_preset(p);
  EXPECT_EXIT({
    EnvMatrixShape sh;
    run_fused_vs_ref_in_child(p, sh);
    std::exit(::testing::Test::HasFailure() ? 1 : 0);
  }, ::testing::ExitedWithCode(0), "");
  unapply_env_preset(p);
}

inline std::string env_case_name(
    const ::testing::TestParamInfo<EnvCase> &info) {
  // gtest test names must be alphanumeric + underscore.  Our labels
  // are already normalised, so just forward.
  return info.param.label;
}

} // namespace

// ── [26] Bucket A: perf-critical env knobs ───────────────────────────────

class TestPrepackEnvBucketA : public ::testing::TestWithParam<EnvCase> {};

TEST_P(TestPrepackEnvBucketA, ChildAssertsCorrectness) {
  run_env_matrix_subprocess_test(GetParam());
}

static std::vector<EnvCase> make_bucket_a_cases() {
  using V = std::vector<std::pair<const char *, const char *>>;
  return {
      // Baseline control: all defaults (no env overrides).  Acts as
      // the regression-floor that every other case is compared
      // against in spirit.
      env_case("baseline_defaults", V{}),

      // ── ZENDNNL_GRP_MATMUL_PREPACK (master switch) ────────────────
      // OFF: the doc-block calls this the "kill-switch for any
      // environment that wants to validate behaves identically with
      // the module compiled out".
      env_case("prepack_off",          V{{"ZENDNNL_GRP_MATMUL_PREPACK", "0"}}),
      env_case("prepack_explicit_on",  V{{"ZENDNNL_GRP_MATMUL_PREPACK", "1"}}),

      // ── ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE ─────────────────────
      // OFF: legacy A/B mode — restores per-call dynamic plan.  Our
      // ALGO 3 AOCL prepack should skip entirely under this knob.
      env_case("stable_ntile_off",          V{{"ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE", "0"}}),
      env_case("stable_ntile_explicit_on",  V{{"ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE", "1"}}),

      // ── ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL (BF16 fast path) ────────
      env_case("custom_kernel_on",           V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"}}),
      env_case("custom_kernel_explicit_off", V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "0"}}),

      // ── ZENDNNL_GRP_MATMUL_ALGO (scheduling) ─────────────────────
      // Each ALGO has its own prepack_for_algo_X body; we exercise
      // all 5 + auto.
      env_case("algo_1", V{{"ZENDNNL_GRP_MATMUL_ALGO", "1"}}),
      env_case("algo_2", V{{"ZENDNNL_GRP_MATMUL_ALGO", "2"}}),
      env_case("algo_3", V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"}}),
      env_case("algo_4", V{{"ZENDNNL_GRP_MATMUL_ALGO", "4"}}),
      env_case("algo_5", V{{"ZENDNNL_GRP_MATMUL_ALGO", "5"}}),

      // ── ZENDNNL_MATMUL_WEIGHT_CACHE ──────────────────────────────
      // Closes B6: AOCL warmer must short-circuit when set to 0.
      // We don't directly observe the AOCL LRU, but the dispatcher
      // call must still produce numerically-correct output.
      env_case("weight_cache_off", V{{"ZENDNNL_MATMUL_WEIGHT_CACHE", "0"}}),

      // ── ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT ───────────────────────
      // OFF: forces wide-dst layout for fused-MoE Op1.  Tests both
      // tight (default) and wide layouts numerically agree.
      env_case("fused_moe_tight_off", V{{"ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT", "0"}}),

      // ── ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT ──────────────────────
      // OFF: ALGO 3 runs fused-swiglu_oai as a separate per-tile
      // pass after the matmul (vs the default in-register fold).
      env_case("n_tile_fused_act_off", V{{"ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT", "0"}}),

      // ── ZENDNNL_GRP_MATMUL_N_ROUNDS (ALGO 3 round picker) ────────
      // The cost-model fix in commit `dac950b7` is THE 4.66% wall-
      // time recovery at 64t; a regression here silently regresses
      // perf.  We force each non-auto value once.
      env_case("n_rounds_force_single",   V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                                            {"ZENDNNL_GRP_MATMUL_N_ROUNDS", "1"}}),
      env_case("n_rounds_force_multi",    V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                                            {"ZENDNNL_GRP_MATMUL_N_ROUNDS", "2"}}),
      env_case("n_rounds_force_balanced", V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                                            {"ZENDNNL_GRP_MATMUL_N_ROUNDS", "3"}}),

      // ── ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR ──────────────────────
      // Pack/microkernel NR.  Cache key includes pack_nr — wrong NR
      // ⇒ silent custom-kernel pack cache miss.
      env_case("custom_kernel_nr_32", V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                                        {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR", "32"}}),
      env_case("custom_kernel_nr_64", V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                                        {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR", "64"}}),
  };
}

INSTANTIATE_TEST_SUITE_P(GrpMatmulEnvBucketA, TestPrepackEnvBucketA,
                         ::testing::ValuesIn(make_bucket_a_cases()),
                         env_case_name);

// ── [27] Bucket B: tuning knobs (smoke + correctness) ────────────────────

class TestPrepackEnvBucketB : public ::testing::TestWithParam<EnvCase> {};

TEST_P(TestPrepackEnvBucketB, ChildAssertsCorrectness) {
  run_env_matrix_subprocess_test(GetParam());
}

static std::vector<EnvCase> make_bucket_b_cases() {
  using V = std::vector<std::pair<const char *, const char *>>;
  return {
      // ── AOCL_TARGET_SLOTS (divisor for stable n_thr) ────────────
      // Default 16; lower → larger stable per expert; raise →
      // smaller.  Per-tile cache-key sizing depends on this.
      env_case("target_slots_8",  V{{"ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS", "8"}}),
      env_case("target_slots_32", V{{"ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS", "32"}}),

      // ── AOCL_BLIS_NC (narrow-N density check input) ──────────────
      env_case("blis_nc_256", V{{"ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC", "256"}}),

      // ── N_ORDER (expert ordering within rounds) ─────────────────
      // Modes 1, 2, 4 only reachable via env (auto picks 0 or 3).
      env_case("n_order_1",   V{{"ZENDNNL_GRP_MATMUL_N_ORDER", "1"}}),
      env_case("n_order_2",   V{{"ZENDNNL_GRP_MATMUL_N_ORDER", "2"}}),
      env_case("n_order_3",   V{{"ZENDNNL_GRP_MATMUL_N_ORDER", "3"}}),
      env_case("n_order_4",   V{{"ZENDNNL_GRP_MATMUL_N_ORDER", "4"}}),

      // ── CUSTOM_KERNEL_SUBTILE_PER_EXPERT ─────────────────────────
      // Doc: "Noise-floor on GPT-OSS decode; may help on large-L2
      // hosts or workloads with extreme M variance."  Smoke-test.
      env_case("subtile_per_expert_on",
               V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT", "1"}}),

      // ── CUSTOM_KERNEL_N_TILE (outer N-tile minimum override) ────
      // Doc: "128 for high threads/num_ops, 512 for prompt-class".
      env_case("custom_n_tile_128",
               V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE", "128"}}),
      env_case("custom_n_tile_512",
               V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE", "512"}}),
  };
}

INSTANTIATE_TEST_SUITE_P(GrpMatmulEnvBucketB, TestPrepackEnvBucketB,
                         ::testing::ValuesIn(make_bucket_b_cases()),
                         env_case_name);

// ── [28] Production-combo interaction matrix ─────────────────────────────

class TestPrepackEnvInteractionMatrix
    : public ::testing::TestWithParam<EnvCase> {};

TEST_P(TestPrepackEnvInteractionMatrix, ChildAssertsCorrectness) {
  run_env_matrix_subprocess_test(GetParam());
}

static std::vector<EnvCase> make_interaction_matrix_cases() {
  using V = std::vector<std::pair<const char *, const char *>>;
  return {
      // Production fast path: ALGO 3 + custom + STABLE_NTILE on.
      env_case("prod_fast_path",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"}}),

      // ALGO 3 AOCL-only (custom off).  Per-tile prepack engaged.
      env_case("algo3_aocl_only",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "0"}}),

      // ALGO 3 + STABLE_NTILE off + custom on.  Custom warm fires;
      // AOCL warm skipped (legacy plan).
      env_case("algo3_stable_off_custom_on",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE", "0"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"}}),

      // ALGO 3 + WEIGHT_CACHE off + custom on.  AOCL warmer
      // short-circuits at entry (B6 fix); custom warm still fires.
      env_case("algo3_weight_cache_off_custom_on",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_MATMUL_WEIGHT_CACHE", "0"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"}}),

      // PREPACK off + everything else on.  Kill-switch — no warming
      // anywhere.  Output must still be correct (fallback to
      // legacy on-demand reorder path).
      env_case("prepack_off_kills_all",
               V{{"ZENDNNL_GRP_MATMUL_PREPACK", "0"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                 {"ZENDNNL_GRP_MATMUL_ALGO", "3"}}),

      // Each non-3 ALGO production path with prepack engaged.
      env_case("algo1_prepack_on",  V{{"ZENDNNL_GRP_MATMUL_ALGO", "1"}}),
      env_case("algo2_prepack_on",  V{{"ZENDNNL_GRP_MATMUL_ALGO", "2"}}),
      env_case("algo4_prepack_on",  V{{"ZENDNNL_GRP_MATMUL_ALGO", "4"}}),
      env_case("algo5_prepack_on",  V{{"ZENDNNL_GRP_MATMUL_ALGO", "5"}}),

      // Auto-ALGO + custom on — let the auto-picker decide while
      // custom kernel pack warm fires.
      env_case("auto_algo_custom_on",
               V{{"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"}}),

      // ALGO 3 + tight off (forces wide path).  Exercises the
      // wide-swiglu correctness guard in flat_n_tile.
      env_case("algo3_wide_layout",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT", "0"}}),

      // ALGO 3 + N_TILE_FUSED_ACT off.  Forces a separate
      // post-matmul activation pass instead of the in-register
      // fold.
      env_case("algo3_unfused_act",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "3"},
                 {"ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT", "0"}}),

      // CROSS_WARM=0 disables the opportunistic cross-regime warm
      // in `cross_warm(...)`.  Closes G4 from the PR-443 review:
      // the env matrix did not previously force the kill-switch
      // for the cross-warm helper.  Correctness must hold (the
      // matmul body runs from the lazy-populated cache instead of
      // the pre-warmed one); a regression here would manifest as
      // an output mismatch in the child subprocess.  Pair with
      // ALGO 1 + CK=1 to exercise the path that cross_warm WOULD
      // populate (custom-kernel regime 3 from an ALGO 1 invocation).
      env_case("algo1_cross_warm_off_custom_on",
               V{{"ZENDNNL_GRP_MATMUL_ALGO", "1"},
                 {"ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL", "1"},
                 {"ZENDNNL_GRP_MATMUL_CROSS_WARM", "0"}}),
  };
}

INSTANTIATE_TEST_SUITE_P(GrpMatmulEnvInteraction,
                         TestPrepackEnvInteractionMatrix,
                         ::testing::ValuesIn(make_interaction_matrix_cases()),
                         env_case_name);

// ===============================================================================
// [29] TestPrepackCrossWarmRegimes - pin the CUSTOM_KERNEL-aware
//      cross-warm decision in `prepack_for_algo_X`.
//
//      Bug-class B12 (deferred decode-time prepack spike):
//      With the auto-select change that routes prompt → ALGO 1, a
//      deployment that fires only prompt during warmup (vLLM) reaches
//      the first decode call with only regime 1 (full-weight AOCL)
//      warm.  The first decode token then pays a ~80-150 ms mid-
//      inference prepack spike for regimes 2 / 3.  The cross-warm
//      logic in `prepack/prepack.cpp::cross_warm(...)` populates the
//      OTHER phase's regime (CUSTOM_KERNEL-aware) on every
//      `prepack_for_algo_X` invocation so the transition is
//      seamless.
//
//      The custom-kernel pack arena exposes per-entry HIT / MISS
//      counters (unlike AOCL DLP, whose probe coalesces both into
//      `packed_ok`), so we use it as the load-bearing observable:
//      after a CK=1 `prepack_for_algo_1`, a custom-kernel probe must
//      report HIT=4 (cross-warm populated all entries); after a CK=0
//      `prepack_for_algo_1`, the same probe must report MISS=4
//      (cross-warm picked regime 2, not regime 3 — custom-kernel
//      cache is untouched).
// ===============================================================================

class TestPrepackCrossWarmRegimes : public ::testing::Test {};

TEST_F(TestPrepackCrossWarmRegimes, CkOnAlgo1CrossWarmsCustomKernelPack) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // CK=1 (set by make_harness default), BF16 dtypes — ck_eligible(p)
  // holds.  `prepack_for_algo_1`'s primary warm populates regime 1
  // (full-weight AOCL); cross-warm should additionally populate
  // regime 3 (custom-kernel pack) because decode will be served by
  // ALGO 3 + custom kernel.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.123f);
  ASSERT_TRUE(h.pp.custom_kernel_on);

  prepack::prepack_for_algo_1(h.pp);

  // Probe the custom-kernel cache: all 4 entries should be HITS
  // because cross-warm populated them.  If cross-warm regressed,
  // they'd be MISSes (the probe itself does the warm).
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits, 4)
      << "CK=1: prepack_for_algo_1 should cross-warm regime 3 "
         "(custom-kernel pack).  If hits<4 the CUSTOM_KERNEL-aware "
         "cross-warm has regressed.";
  EXPECT_EQ(probe.cache_misses, 0);
}

TEST_F(TestPrepackCrossWarmRegimes, CkOffAlgo1DoesNotCrossWarmCustomKernelPack) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // CK=0 → ck_eligible(p) is false → cross-warm picks regime 2
  // (per-tile AOCL DLP), NOT regime 3 (custom-kernel pack).  The
  // custom-kernel cache must stay empty after this call.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.234f);
  h.pp.custom_kernel_on = false;

  prepack::prepack_for_algo_1(h.pp);

  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_misses, 4)
      << "CK=0: cross-warm should pick regime 2 (per-tile AOCL), not "
         "regime 3.  Custom-kernel cache must stay empty.";
  EXPECT_EQ(probe.cache_hits, 0);
}

TEST_F(TestPrepackCrossWarmRegimes,
       CkOnFingerprintDedupAcrossAlgosSharesCustomCache) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // After CK=1 `prepack_for_algo_1` cross-warms regime 3,
  // a subsequent `prepack_for_algo_3` on the same shape:
  //   - has a DIFFERENT fingerprint (scheduling_algo=3 vs 1) so the
  //     process-wide fingerprint cache lets it run, BUT
  //   - finds the custom-kernel cache already populated (cross-warm
  //     did it) → ck_hits should be 4 there too.
  // This locks in the symmetry between the prompt → decode and
  // decode → prompt cross-warm directions.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.345f);
  ASSERT_TRUE(h.pp.custom_kernel_on);

  prepack::prepack_for_algo_1(h.pp);  // primary regime 1 + cross regime 3
  prepack::prepack_for_algo_3(h.pp);  // primary regime 2 + 3 — regime 3 finds hits

  // Probe regime 3 again: still 4 hits (custom-kernel cache is
  // populated once and dedupes across both prepack invocations).
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits, 4);
  EXPECT_EQ(probe.cache_misses, 0);
}

TEST_F(TestPrepackCrossWarmRegimes, CkOnAlgo3CrossWarmsRegime1) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();

  // From ALGO 3, cross-warm should populate regime 1 (full-weight
  // AOCL DLP) — for the case where the framework cycles back to a
  // prompt-class call later in the same process.
  //
  // We can't probe the AOCL DLP LRU's HIT/MISS distribution directly
  // (its probe coalesces both into `packed_ok`).  Instead, we assert
  // that the cross-warm path completes (no allocator / runtime
  // failure) and that the symmetric custom-kernel population still
  // holds via the regime-3 probe — i.e., `prepack_for_algo_3` ran
  // its primary warm correctly.  The regime-1 cross-warm is exercised
  // by the same code path as `warm_aocl(...)` which has its own
  // dedicated regression test in section [23] TestPrepackAoclDlpFullWeight;
  // a leak / crash there would be caught independently.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.456f);
  ASSERT_TRUE(h.pp.custom_kernel_on);
  // Supply num_threads + nr_align so the strict-stable path engages
  // (matches the production ALGO 3 dispatcher's parameterisation).
  h.pp.num_threads = 16;
  h.pp.nr_align    = 1;

  prepack::prepack_for_algo_3(h.pp);

  // Confirm primary regime 3 warm ran (independent of cross-warm).
  prepack::custom_kernel::PackProbeStats probe;
  prepack::custom_kernel::warm_pack_all_custom_kernel_experts(
      h.weight, h.K, h.N, h.ldb, h.transB, h.is_weights_const,
      /*total_count=*/4, probe);
  EXPECT_EQ(probe.cache_hits, 4);
  EXPECT_EQ(probe.cache_misses, 0);
}

// ───────────────────────────────────────────────────────────────────────
// Direct verification of Fix B (skip AOCL DLP per-tile warm under
// CK=1) and Fix D's CK-aware cross-warm routing, via the
// `prepack::test_api::get_last_invocation_stats()` accumulator.
//
// Discriminator: `aocl.total_attempted` field.  It accumulates work
// from BOTH the primary warm and the cross-warm inside
// `prepack_for_algo_X`.  Different scenarios produce predictable
// totals:
//
//   ALGO 1 + CK=1: primary warms regime 1 (full-weight AOCL) at
//     `N` attempts; cross-warm goes to regime 3 (custom-kernel, NOT
//     AOCL).  Total: aocl.total_attempted == N.
//   ALGO 1 + CK=0: primary warms regime 1 at `N`; cross-warm goes
//     to regime 2 (per-tile AOCL with nr_align=1).  Total:
//     aocl.total_attempted == N + N*stable_n_thr.
//   ALGO 3 + CK=1 + Fix B: primary regime 2 SKIPPED (the whole
//     point of Fix B); cross-warm regime 1 (full-weight AOCL) adds
//     `N`.  Total: aocl.total_attempted == N.
//   ALGO 3 + CK=0: primary regime 2 (per-tile) contributes
//     `N*stable_n_thr`; cross-warm regime 1 adds another `N`.
//     Total: aocl.total_attempted == N + N*stable_n_thr.
//
// We pin num_threads = 64 so stable_n_thr = 64/16 = 4 (the production
// default `aocl_stable_n_thr` formula), making the regime-2
// contribution `N*4 = 16` for an N=4 harness — clearly distinguishable
// from the regime-1-only contribution of `N=4`.
// ───────────────────────────────────────────────────────────────────────

TEST_F(TestPrepackCrossWarmRegimes, FixB_CkOnAlgo3SkipsRegime2Warm) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // 4 experts × 64 threads → stable_n_thr = 64/16 = 4 (production
  // formula in `aocl_stable_n_thr`).  If regime 2 were warmed,
  // st_aocl.total_attempted would be N + N*stable = 4 + 16 = 20
  // (per-tile contribution PLUS the cross-warm regime-1
  // contribution).  Fix B drops the per-tile contribution to 0 →
  // total_attempted == 4 (just cross-warm regime 1).
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.111f);
  h.pp.num_threads = 64;
  h.pp.nr_align    = 1;
  ASSERT_TRUE(h.pp.custom_kernel_on);  // make_harness default

  prepack::prepack_for_algo_3(h.pp);

  auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid);
  EXPECT_EQ(stats.scheduling_algo, 3);
  EXPECT_EQ(stats.aocl.total_attempted, 4)
      << "Fix B: under CK=1 + BF16, the per-tile AOCL DLP warm must "
         "be skipped.  Expected `total_attempted == 4` (cross-warm "
         "regime 1 only).  If `> 4`, the per-tile warm has reappeared "
         "and Fix B has regressed.";

  // Sanity: custom-kernel warm did happen (primary regime 3).
  EXPECT_EQ(stats.ck.cache_misses, 4)
      << "Primary regime 3 warm should fire under CK=1 + BF16.";
}

TEST_F(TestPrepackCrossWarmRegimes, FixB_CkOffAlgo3WarmsRegime2) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // Symmetric to the above: under CK=0, `prepack_for_algo_3`'s
  // primary warm IS the per-tile AOCL (regime 2), contributing
  // `N × stable_n_thr = 4 × 4 = 16` attempts.  Cross-warm regime 1
  // is SKIPPED here because the per-tile branch ran (sets
  // primary_did_aocl_fw = false, but cross_warm only skips when
  // primary_did_aocl_fw is true — i.e., the narrow-N escape ran the
  // full-weight warmer).  Wait — re-reading the cross_warm logic:
  // for current_algo == 3, it cross-warms regime 1 unless
  // primary_did_aocl_fw is already true.  Per-tile branch sets
  // primary_did_aocl_fw = false (not the narrow-N escape), so
  // cross-warm fires and adds another N.  Total: 16 + 4 = 20.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.222f);
  h.pp.num_threads = 64;
  h.pp.nr_align    = 1;
  h.pp.custom_kernel_on = false;  // CK=0

  prepack::prepack_for_algo_3(h.pp);

  auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid);
  EXPECT_EQ(stats.scheduling_algo, 3);
  EXPECT_EQ(stats.aocl.total_attempted, 4 + 4 * 4)
      << "CK=0: regime 2 (per-tile, N*stable_n_thr = 16) + cross-warm "
         "regime 1 (full-weight, N = 4) = 20.  If different, either "
         "Fix B over-applied (mistakenly fired under CK=0) or the "
         "cross-warm logic has changed.";

  // Custom-kernel cache should NOT be touched under CK=0.
  EXPECT_EQ(stats.ck.cache_misses, 0);
  EXPECT_EQ(stats.ck.cache_hits, 0);
}

TEST_F(TestPrepackCrossWarmRegimes, FixD_CkOnAlgo1CrossWarmsRegime3OnlyAocl) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // ALGO 1's primary warm is regime 1 (full-weight AOCL): N=4
  // attempts.  Under CK=1 cross-warm targets regime 3 (custom-kernel
  // pack), NOT regime 2 — so `aocl.total_attempted` reflects only
  // the primary regime-1 contribution (= 4) and `ck.cache_misses`
  // shows the cross-warm regime-3 contribution.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.333f);
  h.pp.num_threads = 64;
  h.pp.nr_align    = 1;
  ASSERT_TRUE(h.pp.custom_kernel_on);

  prepack::prepack_for_algo_1(h.pp);

  auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid);
  EXPECT_EQ(stats.scheduling_algo, 1);
  EXPECT_EQ(stats.aocl.total_attempted, 4)
      << "CK=1 + ALGO 1: cross-warm picks regime 3, not regime 2 — "
         "aocl.total_attempted should be N (primary regime 1 only).";
  EXPECT_EQ(stats.ck.cache_misses, 4)
      << "CK=1 + ALGO 1: cross-warm should populate regime 3 "
         "(custom-kernel pack) with N entries.";
}

TEST_F(TestPrepackCrossWarmRegimes, FixD_CkOffAlgo1CrossWarmsRegime2) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // ALGO 1's primary regime-1 warm contributes N=4 attempts.  Under
  // CK=0 cross-warm picks regime 2 (per-tile AOCL, nr_align=1) — adds
  // `N × stable_n_thr = 4 × 4 = 16` attempts.  Total: 20.  Custom-
  // kernel cache untouched.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.444f);
  h.pp.num_threads = 64;
  h.pp.nr_align    = 1;
  h.pp.custom_kernel_on = false;  // CK=0

  prepack::prepack_for_algo_1(h.pp);

  auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid);
  EXPECT_EQ(stats.scheduling_algo, 1);
  EXPECT_EQ(stats.aocl.total_attempted, 4 + 4 * 4)
      << "CK=0 + ALGO 1: cross-warm should populate regime 2 — "
         "expected N (primary regime 1) + N*stable_n_thr (cross-warm "
         "regime 2) = 4 + 16 = 20.";
  EXPECT_EQ(stats.ck.cache_misses, 0);
}

TEST_F(TestPrepackCrossWarmRegimes, CrossWarmOnAlgo1NoCustomMatchesPrimaryOnly) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // Edge case: CK=0 + missing thread context (`num_threads == 0`)
  // disables the regime-2 cross-warm safely (since stable n_tile
  // can't be computed without thread context).  The primary regime-1
  // warm still runs.  Expected: aocl.total_attempted == N from
  // primary only; cross-warm short-circuits cleanly.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.555f);
  h.pp.custom_kernel_on = false;
  h.pp.num_threads      = 0;   // missing thread context
  h.pp.nr_align         = 0;

  prepack::prepack_for_algo_1(h.pp);

  auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid);
  EXPECT_EQ(stats.scheduling_algo, 1);
  EXPECT_EQ(stats.aocl.total_attempted, 4)
      << "ALGO 1 primary regime-1 warm should run unconditionally on "
         "AOCL DLP path; cross-warm regime 2 should safely skip when "
         "thread context is missing.";
}

// ===============================================================================
// [30] TestPrepackFingerprintInvariance - pin the order-independent
//      fingerprint behaviour added to address Copilot review
//      comment #1 on PR-443.  Two flavours:
//
//        (a) Permuting the weight pointer pool MUST keep the
//            fingerprint stable (same set, same hash) so a caller
//            that rotates the active subset inside a stable expert
//            pool short-circuits on `already_warmed`.
//        (b) Replacing a member of the pool (different set) MUST
//            change the fingerprint so a real expert-table swap
//            re-fires the warm.
//
//      Both observable through `test_api::LastInvocationStats`:
//      first call populates; clear stats; second call leaves
//      stats untouched (`!valid`) IFF the prelude short-circuited
//      via the fingerprint.
// ===============================================================================

class TestPrepackFingerprintInvariance : public ::testing::Test {};

TEST_F(TestPrepackFingerprintInvariance, PermutationDoesNotRefireWarm) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // First call: populates the fingerprint cache and the test_api
  // accumulator.  Use ALGO 1 + AOCL DLP path so the fingerprint
  // covers the AOCL warm path too.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.111f);
  h.pp.custom_kernel_on = false;
  h.pp.num_threads      = 1;
  h.pp.nr_align         = 1;
  prepack::prepack_for_algo_1(h.pp);
  ASSERT_TRUE(prepack::test_api::get_last_invocation_stats().valid)
      << "first call must populate the test_api accumulator";

  // Permute the pointer pool in place — same set of pointers, just
  // reordered.  Under the order-independent (XOR) fingerprint this
  // MUST keep the hash stable; under the previous 3-sample hash the
  // permutation flipped weight[0] / weight[n-1] and forced a refire.
  std::swap(h.weight[0], h.weight[3]);
  std::swap(h.weight[1], h.weight[2]);

  prepack::test_api::clear_last_invocation_stats();
  prepack::prepack_for_algo_1(h.pp);

  // Second call should hit `already_warmed` in the prelude and
  // short-circuit before reaching `log_pack_probe`, leaving the
  // accumulator un-populated.
  const auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_FALSE(stats.valid)
      << "permuting the pointer pool with the same set must NOT "
         "re-fire the warm — fingerprint is order-independent (XOR "
         "across the full pool).  scheduling_algo="
      << stats.scheduling_algo;
}

TEST_F(TestPrepackFingerprintInvariance, MembershipChangeRefireWarm) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.222f);
  h.pp.custom_kernel_on = false;
  h.pp.num_threads      = 1;
  h.pp.nr_align         = 1;
  prepack::prepack_for_algo_1(h.pp);
  ASSERT_TRUE(prepack::test_api::get_last_invocation_stats().valid);

  // Swap one entry for a brand-new buffer — different SET means the
  // fingerprint MUST flip, so the next call re-fires the warm and
  // populates the accumulator again.  Use a separate bank to ensure
  // the new pointer is not part of the old pool by chance.
  std::vector<bfloat16_t> fresh_bank((size_t)32 * 64, bfloat16_t(0.999f));
  h.weight[2] = fresh_bank.data();

  prepack::test_api::clear_last_invocation_stats();
  prepack::prepack_for_algo_1(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid)
      << "replacing a pool member with a new pointer must re-fire "
         "the warm — fingerprint is sensitive to set membership.";
  EXPECT_EQ(stats.scheduling_algo, 1);
}

TEST_F(TestPrepackFingerprintInvariance, PoolSizeChangeRefireWarm) {
  using namespace zendnnl::lowoha::matmul;
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;

  reset_grp_matmul_caches();
  prepack::test_api::clear_last_invocation_stats();

  // Pathological case: an attacker-style XOR collision (e.g. all
  // pointers equal so the XOR is 0) would otherwise pass the pool-XOR
  // alone.  The fingerprint folds `weight->size()` separately, so a
  // pool-size change MUST refire.  This test guards against future
  // regressions that might drop the size term.
  auto h_small = make_harness(/*total=*/4, /*active=*/4,
                              /*K=*/32, /*N=*/64, /*fill=*/0.333f);
  h_small.pp.custom_kernel_on = false;
  h_small.pp.num_threads      = 1;
  h_small.pp.nr_align         = 1;
  prepack::prepack_for_algo_1(h_small.pp);
  ASSERT_TRUE(prepack::test_api::get_last_invocation_stats().valid);

  auto h_big = make_harness(/*total=*/8, /*active=*/8,
                            /*K=*/32, /*N=*/64, /*fill=*/0.333f);
  h_big.pp.custom_kernel_on = false;
  h_big.pp.num_threads      = 1;
  h_big.pp.nr_align         = 1;

  prepack::test_api::clear_last_invocation_stats();
  prepack::prepack_for_algo_1(h_big.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid)
      << "pool size change (4 -> 8) must re-fire the warm even when "
         "every pointer differs — fingerprint folds weight->size() "
         "separately to prevent XOR-collision false-hits.";
}

// ===============================================================================
// [32] TestPrepackCkGateSymmetry - pin the prepack-runtime CK gate
//      symmetry contract.
//
//      Bug-class B17 (asymmetric prepack vs runtime CK refusal):
//      Prepack `ck_eligible(p)` historically only checked the bf16
//      dtype trio + `custom_kernel_on`, but the runtime gate
//      (`custom_kernel/dispatch.cpp::prepare_for_call`) refuses CK
//      on 11 distinct grounds.  When the runtime refused for any
//      reason prepack didn't mirror, the warm-pack module would:
//        * populate the CK pack arena (`warm_custom`) — wasted,
//          since the runtime never reads it
//        * skip the per-tile AOCL warm via Fix B's `!ck_eligible`
//          guard — since prepack thought CK would engage
//      Result at gpt-oss-20B class: ~1.5 GB wasted CK arena + ~12 k
//      lazy AOCL DLP reorders at first execution.  Concretely, the
//      production trigger is fused-MoE with `silu_and_mul` or
//      `gelu_and_mul` — both refused by the runtime CK (only
//      `swiglu_oai_mul` / `none` are accepted) but both passed
//      prepack's old gate.
//
//      Tightened gates (prepack now mirrors):
//        * activation in {swiglu_oai_mul, none}
//        * act_dtype = bf16 when act != none
//        * bias_dtype in {none, bf16, f32}
//        * pack_nr in {32, 64} divides representative N
//
//      Each TEST_F drives `prepack_for_algo_3` with one gate flipped
//      in isolation, then asserts via the test_api accumulator that
//      no CK warm fired (= ck_eligible returned false) for the
//      refusal cases, and that all-good config does fire CK warm.
// ===============================================================================

class TestPrepackCkGateSymmetry : public ::testing::Test {};

TEST_F(TestPrepackCkGateSymmetry, AllGatesPassEnablesCustomKernelWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // Baseline: BF16/BF16/BF16 + CK on + act = swiglu_oai_mul (accepted)
  // + act_dtype = bf16 + bias_dtype = none + N=64 (divides 32).
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.401f);
  h.pp.act       = grp_matmul_gated_act_t::swiglu_oai_mul;
  h.pp.act_dtype = data_type_t::bf16;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_GT(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "All-pass gate must engage warm_custom (CK pack arena populated)";
}

TEST_F(TestPrepackCkGateSymmetry, ActSiluRefusedNoCkWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // silu_and_mul is refused by the runtime CK (only swiglu_oai_mul /
  // none are accepted).  Prepack must mirror -> no CK warm.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.402f);
  h.pp.act       = grp_matmul_gated_act_t::silu_and_mul;
  h.pp.act_dtype = data_type_t::bf16;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "act=silu_and_mul is refused by runtime CK; prepack must NOT "
         "warm CK pack arena (wasted memory + Fix-B skipped per-tile "
         "AOCL warm + runtime falls back to AOCL DLP per-tile)";
}

TEST_F(TestPrepackCkGateSymmetry, ActGeluRefusedNoCkWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.403f);
  h.pp.act       = grp_matmul_gated_act_t::gelu_and_mul;
  h.pp.act_dtype = data_type_t::bf16;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "act=gelu_and_mul is refused by runtime CK; prepack must NOT "
         "warm CK pack arena";
}

TEST_F(TestPrepackCkGateSymmetry, ActDtypeF32WithGatedActRefusedNoCkWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // swiglu_oai_mul + act_dtype = f32 -> runtime CK refuses
  // (act_dtype must be bf16 when act != none).
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.404f);
  h.pp.act       = grp_matmul_gated_act_t::swiglu_oai_mul;
  h.pp.act_dtype = data_type_t::f32;  // refused
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "act_dtype=f32 with gated act is refused by runtime CK; "
         "prepack must NOT warm CK pack arena";
}

TEST_F(TestPrepackCkGateSymmetry, ActNoneIgnoresActDtype) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // act = none + act_dtype = f32 (or anything) -> runtime CK accepts
  // because act_dtype is ignored when act == none.  Prepack must
  // also accept.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.405f);
  h.pp.act       = grp_matmul_gated_act_t::none;
  h.pp.act_dtype = data_type_t::f32;  // ignored when act=none
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_GT(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "act=none ignores act_dtype; CK must engage";
}

TEST_F(TestPrepackCkGateSymmetry, BiasDtypeF16RefusedNoCkWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // bias_dtype = f16 -> runtime CK refuses (only none / bf16 / f32).
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.406f);
  h.pp.act        = grp_matmul_gated_act_t::none;
  h.pp.bias_dtype = data_type_t::f16;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "bias_dtype=f16 is refused by runtime CK; prepack must NOT "
         "warm CK pack arena";
}

TEST_F(TestPrepackCkGateSymmetry, BiasDtypeBf16AndF32Accepted) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;

  for (data_type_t bias_dt : {data_type_t::none, data_type_t::bf16,
                              data_type_t::f32}) {
    clear_custom_kernel_pack_cache();
    prepack::clear_fingerprint_cache_for_test();
    prepack::test_api::clear_last_invocation_stats();

    auto h = make_harness(/*total=*/4, /*active=*/4,
                          /*K=*/32, /*N=*/64, /*fill=*/0.407f);
    h.pp.act        = grp_matmul_gated_act_t::none;
    h.pp.bias_dtype = bias_dt;
    h.pp.num_threads = 1;
    h.pp.nr_align    = 1;
    prepack::prepack_for_algo_3(h.pp);

    const auto stats = prepack::test_api::get_last_invocation_stats();
    ASSERT_TRUE(stats.valid);
    EXPECT_GT(stats.ck.cache_misses + stats.ck.cache_hits, 0)
        << "bias_dtype " << static_cast<int>(bias_dt) << " is accepted by "
           "runtime CK; prepack must engage CK warm";
  }
}

TEST_F(TestPrepackCkGateSymmetry, NNotMultipleOfPackNrRefusedNoCkWarm) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // N=48 doesn't divide 32 (48%32=16) and doesn't divide 64 (48%64=48)
  // -> runtime CK refuses.  Prepack must mirror.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/48, /*fill=*/0.408f);
  h.pp.act = grp_matmul_gated_act_t::none;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.ck.cache_misses + stats.ck.cache_hits, 0)
      << "N=48 not divisible by 32 or 64 -> runtime CK refuses (pack_nr "
         "planner returns 0); prepack must NOT warm CK pack arena";
}

TEST_F(TestPrepackCkGateSymmetry, FingerprintReDispatchesOnActChange) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache;
  clear_custom_kernel_pack_cache();
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  // First call with act=swiglu_oai_mul -> CK eligible -> warm fires.
  auto h = make_harness(/*total=*/4, /*active=*/4,
                        /*K=*/32, /*N=*/64, /*fill=*/0.409f);
  h.pp.act       = grp_matmul_gated_act_t::swiglu_oai_mul;
  h.pp.act_dtype = data_type_t::bf16;
  h.pp.num_threads = 1;
  h.pp.nr_align    = 1;
  prepack::prepack_for_algo_3(h.pp);
  ASSERT_TRUE(prepack::test_api::get_last_invocation_stats().valid);

  // Same pool, same scheduling algo, but act flipped to silu_and_mul:
  // ck_eligible flips false; without the act fingerprint term, the
  // second call would hit `already_warmed` and short-circuit, leaving
  // the (false) verdict cached.  With the fingerprint term, second
  // call re-fires and arrives at the new (refused) verdict.
  prepack::test_api::clear_last_invocation_stats();
  h.pp.act = grp_matmul_gated_act_t::silu_and_mul;
  prepack::prepack_for_algo_3(h.pp);

  const auto stats = prepack::test_api::get_last_invocation_stats();
  EXPECT_TRUE(stats.valid)
      << "Activation change must re-fire the prepack so ck_eligible "
         "is re-evaluated against the current call's act";
  // After the act flip, no new CK warm should fire (silu refused).
  // Counters are accumulators across the second call only.
  EXPECT_EQ(stats.ck.cache_misses, 0)
      << "Second call (silu refused) must NOT warm CK arena";
}

// ===============================================================================
// [33] TestPrepackBuildParamsContract - pin `build_prepack_params` to
//      the dispatcher's active/total contract
//      (`group_matmul_direct.cpp:203-214`).  Specifically:
//
//        * active_matmul == 0, total_matmul == 0 (legacy):
//            num_ops_active = M.size()
//            num_ops_total  = num_ops_active
//        * active_matmul == 0, total_matmul > 0 (legacy + stale total):
//            num_ops_active = M.size()
//            num_ops_total  = num_ops_active
//          (total_matmul ignored — only meaningful inside opt-in,
//           mirrors the dispatcher's `framework_opt_in` gate)
//        * active_matmul > 0, total_matmul == 0:
//            num_ops_active = active_matmul
//            num_ops_total  = active_matmul
//        * active_matmul > 0, total_matmul > 0:
//            num_ops_active = active_matmul
//            num_ops_total  = total_matmul
//
//      The dispatcher accepts both Compact (`M.size() == active_matmul`)
//      and Padded (`M.size() == total_matmul`) input layouts.  The
//      Padded case is what regressed pre-fix: `build_prepack_params`
//      used `M.size()` for `num_ops_active`, over-counting the firing
//      experts in the PACK_PROBE log and any downstream diagnostic.
//      Each TEST_F pins one row of the contract.
// ===============================================================================

class TestPrepackBuildParamsContract : public ::testing::Test {};

namespace {
// Local helper — builds the minimum weight/K/N/ldb/transB/is_const
// vectors needed to call `build_prepack_params`.  All sized to
// `pool_n` and filled with placeholder values; the test only reads
// the returned `PrepackParams`' `num_ops_active` / `num_ops_total`.
struct BuildParamsFixture {
  std::vector<float>        bank_storage;  // single backing buffer
  std::vector<const void *> weight;
  std::vector<int>          K;
  std::vector<int>          N;
  std::vector<int>          ldb;
  std::vector<bool>         transB;
  std::vector<bool>         is_const;
};

inline BuildParamsFixture make_build_fixture(int pool_n) {
  BuildParamsFixture f;
  f.bank_storage.assign(static_cast<size_t>(pool_n) * 4, 0.0f);
  f.weight.resize(pool_n);
  for (int i = 0; i < pool_n; ++i) {
    f.weight[i] = &f.bank_storage[static_cast<size_t>(i) * 4];
  }
  f.K       .assign(pool_n, 4);
  f.N       .assign(pool_n, 32);
  f.ldb     .assign(pool_n, 32);
  f.transB  .assign(pool_n, false);
  f.is_const.assign(pool_n, true);
  return f;
}
}  // namespace

TEST_F(TestPrepackBuildParamsContract, LegacyBothZeroResolvesToMSize) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Legacy contract: caller leaves both fields at default 0.
  // Expected: num_ops_active = num_ops_total = M.size().
  auto f = make_build_fixture(/*pool_n=*/4);
  std::vector<matmul_params> params(1);   // both fields default-zero
  std::vector<int>           M(4, 1);     // legacy: every entry of M fires

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 4);
  EXPECT_EQ(pp.num_ops_total,  4);
}

TEST_F(TestPrepackBuildParamsContract, LegacyStaleTotalMatmulIsIgnored) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Anomalous-but-possible: a legacy caller leaves `active_matmul = 0`
  // (no opt-in) but, through stale memory or copy-paste from an opt-in
  // caller, has `total_matmul > 0`.  The dispatcher's `framework_opt_in`
  // is purely `active_matmul > 0`, so it treats this call as legacy
  // (`num_ops = M.size()`) and the size validator strictly requires
  // every per-expert vector to be exactly `M.size()` long.
  //
  // `build_prepack_params` must mirror that gate exactly — honouring
  // the stale `total_matmul` here would:
  //   - create a distinct fingerprint cache entry vs the equivalent
  //     legacy call with total_matmul=0 (the fingerprint hash takes
  //     `num_ops_total` as its first input);
  //   - have the warmer iterate `[0, total_matmul)` over weight
  //     vectors that strict legacy size validation requires to be
  //     exactly `M.size()` long (no UB — the warmer's `min({...})`
  //     clamp still bounds the loop — but the PACK_PROBE log line
  //     would report a misleading `total=N` larger than the count the
  //     dispatcher actually runs).
  // Expected: both fields fall back to `M.size()`.
  auto f = make_build_fixture(/*pool_n=*/4);
  std::vector<matmul_params> params(1);
  params[0].active_matmul = 0;            // legacy: no opt-in
  params[0].total_matmul  = 8;            // stale: must be ignored
  std::vector<int>           M(4, 1);

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 4)
      << "Legacy active=0 must fall back to M.size()=4";
  EXPECT_EQ(pp.num_ops_total,  4)
      << "Stale total_matmul=8 must be ignored when active_matmul=0 "
         "(mirrors group_matmul_direct.cpp::framework_opt_in gate)";
}

TEST_F(TestPrepackBuildParamsContract, ActiveOnlyCompactHonorsActive) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Active hint only, Compact input: M.size() == active_matmul.
  // total_matmul = 0 means "no prepack-extras tail" -> warm only the
  // firing experts.
  auto f = make_build_fixture(/*pool_n=*/3);
  std::vector<matmul_params> params(1);
  params[0].active_matmul = 3;
  params[0].total_matmul  = 0;
  std::vector<int>           M(3, 1);

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 3);
  EXPECT_EQ(pp.num_ops_total,  3);
}

TEST_F(TestPrepackBuildParamsContract, ActiveOnlyPaddedHonorsActiveNotMSize) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // The behaviour-fix case.  Active hint only, Padded input:
  // M.size() == 8 with M[3..8) = 0 placeholders; only experts
  // [0..3) fire.  Pre-fix: num_ops_active = M.size() = 8 (wrong).
  // Post-fix: num_ops_active = active_matmul = 3.
  auto f = make_build_fixture(/*pool_n=*/8);
  std::vector<matmul_params> params(1);
  params[0].active_matmul = 3;
  params[0].total_matmul  = 0;     // no prepack-extras tail
  std::vector<int>           M(8, 0);
  M[0] = M[1] = M[2] = 1;          // firing experts only

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 3)
      << "Padded form: num_ops_active must reflect active_matmul=3, "
         "not M.size()=8 (pre-fix bug)";
  EXPECT_EQ(pp.num_ops_total,  3)
      << "total_matmul=0 -> num_ops_total falls back to num_ops_active=3";
}

TEST_F(TestPrepackBuildParamsContract, FullHintCompactWarmsTotalPool) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Full framework hint, Compact input: M.size() == active_matmul = 3,
  // total_matmul = 8 (prepack-extras tail).
  auto f = make_build_fixture(/*pool_n=*/8);
  std::vector<matmul_params> params(1);
  params[0].active_matmul = 3;
  params[0].total_matmul  = 8;
  std::vector<int>           M(3, 1);

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 3);
  EXPECT_EQ(pp.num_ops_total,  8)
      << "total_matmul>0 must drive the prepack-extras tail warm-up";
}

TEST_F(TestPrepackBuildParamsContract, FullHintPaddedHonorsActiveAndTotal) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Full framework hint, Padded input: M.size() == total_matmul = 8,
  // active_matmul = 3 (firing experts), M[3..8) = 0 placeholders.
  // Both fields populated -> active reflects active_matmul, total
  // reflects total_matmul.
  auto f = make_build_fixture(/*pool_n=*/8);
  std::vector<matmul_params> params(1);
  params[0].active_matmul = 3;
  params[0].total_matmul  = 8;
  std::vector<int>           M(8, 0);
  M[0] = M[1] = M[2] = 1;

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 3)
      << "Padded form with both fields set: num_ops_active reflects "
         "active_matmul=3, not M.size()=8";
  EXPECT_EQ(pp.num_ops_total,  8);
}

TEST_F(TestPrepackBuildParamsContract, EmptyParamsFallsBackToMSize) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  using zendnnl::lowoha::matmul::matmul_params;

  // Edge case: caller passes an empty `params` vector (legacy callers
  // that don't construct params at all).  Both fields effectively 0 -
  // we fall through to M.size().
  auto f = make_build_fixture(/*pool_n=*/5);
  std::vector<matmul_params> params;   // EMPTY
  std::vector<int>           M(5, 1);

  auto pp = prepack::build_prepack_params(
      f.weight, f.K, f.N, f.ldb, f.transB, f.is_const,
      params, M, /*custom_kernel_on=*/false);

  EXPECT_EQ(pp.num_ops_active, 5);
  EXPECT_EQ(pp.num_ops_total,  5);
}
