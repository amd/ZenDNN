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
#include "operators/matmul/matmul_config.hpp"

namespace {

namespace ck = ck_test::ck;
namespace mt = moe_test_utils;

// RAII guard for the library-wide weight-cache toggle.  Save the
// current `matmul_config_t::get_weight_cache()` value at construction,
// set the requested value, restore at destruction.  Allows individual
// tests to flip the toggle without leaking state into sibling tests
// running in the same process.  Mirrors `mt::CustomKernelNROverride`.
class WeightCacheOverride {
 public:
  explicit WeightCacheOverride(int32_t value)
      : prev_(zendnnl::ops::matmul_config_t::instance().get_weight_cache()) {
    zendnnl::ops::matmul_config_t::instance().set_weight_cache(value);
  }
  ~WeightCacheOverride() {
    zendnnl::ops::matmul_config_t::instance().set_weight_cache(prev_);
  }
  WeightCacheOverride(const WeightCacheOverride &)            = delete;
  WeightCacheOverride &operator=(const WeightCacheOverride &) = delete;

 private:
  int32_t prev_;
};

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

// ──────────────────────────────────────────────────────────────────
// silu_and_mul / gelu_and_mul interleave layout —
//   bit-equality with swiglu_oai_mul.
//
// Both fused split-halves paths (silu and gelu) are built on this
// physical invariant: after the prepack permutes a canonical
// split-halves W13 `[gate_cols | up_cols]` into the CK arena, the
// resulting bytes MUST be bit-identical to what `swiglu_oai_mul`
// produces when fed an already-interleaved W13
// `[g0, u0, g1, u1, ...]`.  silu and gelu use the SAME permutation
// (they differ only in the kernel-side activation math, not in the
// pack layout), so this test asserts both lay down the same bytes.
//
// Construction:
//   * Build an interleaved weight `W_i[k, 2*j+0] = gate[k, j]`,
//                                  `W_i[k, 2*j+1] = up  [k, j]`.
//   * Build a split-halves weight `W_s[k, j]       = gate[k, j]`,
//                                  `W_s[k, I + j]   = up  [k, j]`.
//   * Pack `W_i` with `act = swiglu_oai_mul` (no interleave flag).
//   * Pack `W_s` with `act = silu_and_mul` (interleave flag set).
//   * Pack `W_s` with `act = gelu_and_mul` (interleave flag set —
//     same physical layout as silu, since the cache key shares
//     `kInterleaveSplitMarker` and the prepack permutation is
//     activation-agnostic).
//   * Assert all three packed buffers are byte-identical.
//
// We drive every pack via the public `prepare_for_call` path so the
// test exercises the same code that runs in production.
//
// Cache-key note: silu and gelu share the same `interleave=1` cache
// key bit, so technically the second of the two prepares (with the
// same weight pointer + dims + transB + pack_nr) would HIT the same
// LRU entry the first one populated.  To exercise the actual gelu
// pack code path (and not just a HIT against silu's earlier entry)
// we use distinct weight buffers for silu vs gelu.
// ──────────────────────────────────────────────────────────────────
TEST(CkPackBf16, SiluGeluInterleavedPackMatchesSwigluBytes) {
  CK_SKIP_IF_NO_BF16_ISA();
  ::reset_grp_matmul_caches();

  constexpr int kK = 64;
  constexpr int kN = 256;
  constexpr int kI = kN / 2;

  // Deterministic source data — same value generator across all
  // three logical weight buffers so the only physical difference is
  // row order.
  auto val_gate = [](int k, int j) {
    return static_cast<float>(k * 31 + j) * 1.0e-3f;
  };
  auto val_up = [](int k, int j) {
    return static_cast<float>(k * 31 + j + kI) * 1.0e-3f;
  };

  std::vector<bfloat16_t> w_interleaved(
      static_cast<size_t>(kK) * kN, bfloat16_t(0.0f));
  // Two distinct split-halves buffers — same logical content.
  // Distinct pointers force distinct LRU keys so the second pack
  // (gelu) actually runs the prepack path instead of HITting the
  // entry the first pack (silu) populated.
  std::vector<bfloat16_t> w_split_silu(
      static_cast<size_t>(kK) * kN, bfloat16_t(0.0f));
  std::vector<bfloat16_t> w_split_gelu(
      static_cast<size_t>(kK) * kN, bfloat16_t(0.0f));
  for (int k = 0; k < kK; ++k) {
    for (int j = 0; j < kI; ++j) {
      w_interleaved[k * kN + 2 * j + 0] = bfloat16_t(val_gate(k, j));
      w_interleaved[k * kN + 2 * j + 1] = bfloat16_t(val_up  (k, j));
      w_split_silu [k * kN + j]         = bfloat16_t(val_gate(k, j));
      w_split_silu [k * kN + kI + j]    = bfloat16_t(val_up  (k, j));
      w_split_gelu [k * kN + j]         = bfloat16_t(val_gate(k, j));
      w_split_gelu [k * kN + kI + j]    = bfloat16_t(val_up  (k, j));
    }
  }

  // Pack all three layouts via prepare_for_call.
  std::vector<bool>  transA_v{false};
  std::vector<bool>  transB_v{false};
  std::vector<int>   M_v{16};
  std::vector<int>   N_v{kN};
  std::vector<int>   K_v{kK};
  std::vector<int>   ldb_v{kN};
  std::vector<float> alpha_v{1.0f};
  std::vector<float> beta_v{0.0f};
  std::vector<bool>  is_wc_v{true};

  std::vector<const void *> wi_v{w_interleaved.data()};
  std::vector<const void *> ws_silu_v{w_split_silu.data()};
  std::vector<const void *> ws_gelu_v{w_split_gelu.data()};

  ck::CallContext kctx_swiglu, kctx_silu, kctx_gelu;
  ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::swiglu_oai_mul,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::none,
                                  transA_v, transB_v, M_v, N_v, K_v, ldb_v,
                                  alpha_v, beta_v, wi_v, is_wc_v,
                                  kctx_swiglu),
            zendnnl::error_handling::status_t::success);
  ASSERT_TRUE(kctx_swiglu.enabled);

  ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::silu_and_mul,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::none,
                                  transA_v, transB_v, M_v, N_v, K_v, ldb_v,
                                  alpha_v, beta_v, ws_silu_v, is_wc_v,
                                  kctx_silu),
            zendnnl::error_handling::status_t::success);
  ASSERT_TRUE(kctx_silu.enabled);

  ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::gelu_and_mul,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::bf16, data_type_t::bf16,
                                  data_type_t::none,
                                  transA_v, transB_v, M_v, N_v, K_v, ldb_v,
                                  alpha_v, beta_v, ws_gelu_v, is_wc_v,
                                  kctx_gelu),
            zendnnl::error_handling::status_t::success);
  ASSERT_TRUE(kctx_gelu.enabled);

  // Same `pack_nr` chosen by all three paths (planner is a pure
  // function of (K, N)).
  ASSERT_EQ(kctx_swiglu.pack_nr, kctx_silu.pack_nr);
  ASSERT_EQ(kctx_swiglu.pack_nr, kctx_gelu.pack_nr);

  const int pack_nr = kctx_swiglu.pack_nr;
  const int K_pair = (kK + 1) / 2;
  const size_t pack_bytes = static_cast<size_t>(kN / pack_nr)
      * K_pair * pack_nr * 2 /*VNNI pair*/ * sizeof(bfloat16_t);

  const auto *p_swiglu = static_cast<const bfloat16_t *>(
      kctx_swiglu.packed_ptrs[0]);
  const auto *p_silu = static_cast<const bfloat16_t *>(
      kctx_silu.packed_ptrs[0]);
  const auto *p_gelu = static_cast<const bfloat16_t *>(
      kctx_gelu.packed_ptrs[0]);
  ASSERT_NE(p_swiglu, nullptr);
  ASSERT_NE(p_silu, nullptr);
  ASSERT_NE(p_gelu, nullptr);

  // Bit-equality — the silu/gelu prepack column permutation must be
  // the exact inverse of the caller's split-halves layout, so the
  // packed arena lands at the same physical bytes the swiglu path
  // produces from a pre-interleaved input.
  EXPECT_EQ(0, std::memcmp(p_swiglu, p_silu, pack_bytes))
      << "silu_and_mul pack bytes do not match swiglu_oai_mul pack "
         "bytes — the silu in-register fused epilogue would "
         "deinterleave (g, u) from the wrong columns and produce "
         "silent-wrong activations.";
  EXPECT_EQ(0, std::memcmp(p_swiglu, p_gelu, pack_bytes))
      << "gelu_and_mul pack bytes do not match swiglu_oai_mul pack "
         "bytes — the gelu in-register fused epilogue would "
         "deinterleave (g, u) from the wrong columns and produce "
         "silent-wrong activations.  silu and gelu MUST share the "
         "same prepack permutation (only the kernel-side activation "
         "math differs).";
  EXPECT_EQ(0, std::memcmp(p_silu, p_gelu, pack_bytes))
      << "silu_and_mul and gelu_and_mul packs differ — they should "
         "be byte-identical for the same logical weight, since the "
         "prepack interleave is activation-agnostic.";
}

// ──────────────────────────────────────────────────────────────────
// ZENDNNL_MATMUL_WEIGHT_CACHE=0 (no-cache mode) contract:
// Under `matmul_config_t::set_weight_cache(0)` (equivalent to the
// env knob `ZENDNNL_MATMUL_WEIGHT_CACHE=0`) the runtime CK path
// routes every per-expert pack through
// `get_or_pack_weight_bf16(..., disable_cache=true)`, which
// allocates a fresh aligned buffer per call without touching the
// LRU singleton.  The buffers are owned by the `CallContext` and
// freed by its destructor (or `release_owned_buffers()` /
// `reset()`).  This suite pins the three invariants of that mode:
//
//   1) CK still ENGAGES (no refuse → no DLP fallback).
//   2) Two prepares with the SAME (weight ptr, K, N, ldb, transB)
//      produce DISTINCT packed pointers (the cache is bypassed —
//      the second call cannot hit the first's entry because the
//      first never inserted into the LRU).
//   3) `owned_packed_ptrs[i]` aliases `packed_ptrs[i]` for every
//      active expert (the dispatcher's contract for the destructor
//      free path).
//
// These invariants are mutually exclusive with the
// `SecondPrepareHitsCacheSamePackedPtr` test above, which asserts
// the OPPOSITE contract under the default mode — covering the
// flip-side explicitly here documents the design intent that the
// two modes are functionally distinct, not just runtime-faster
// vs slower variants of the same caching policy.
// ──────────────────────────────────────────────────────────────────
TEST(CkPackBf16NoCache, CkEngagesAndAllocatesCallerOwnedPacks) {
  CK_SKIP_IF_NO_BF16_ISA();
  ::reset_grp_matmul_caches();
  WeightCacheOverride wc_off(0);

  ck_test::PrepCallCase c{};
  c.label = "ck_engages_under_weight_cache_zero";

  ck_test::PrepCallStorage storage;
  ck::CallContext kctx;
  ASSERT_EQ(ck_test::run_prepare(c, storage, kctx),
            zendnnl::error_handling::status_t::success)
      << "prepare_for_call must succeed under WEIGHT_CACHE=0 — the "
         "no-cache mode is supposed to switch the per-expert pack "
         "to caller-owned buffers, NOT refuse CK entirely.";
  EXPECT_TRUE(kctx.enabled)
      << "kctx.enabled must remain true under WEIGHT_CACHE=0 "
         "(no CK refusal — the runtime keeps packing, just without "
         "the LRU singleton)";
  EXPECT_NE(kctx.packed_ptrs[0], nullptr);
  EXPECT_NE(kctx.owned_packed_ptrs[0], nullptr)
      << "owned_packed_ptrs[0] must be populated under "
         "WEIGHT_CACHE=0 so the CallContext destructor knows to "
         "free the caller-owned buffer.";
  EXPECT_EQ(static_cast<const void *>(kctx.packed_ptrs[0]),
            static_cast<const void *>(kctx.owned_packed_ptrs[0]))
      << "packed_ptrs[0] must alias owned_packed_ptrs[0] in no-cache "
         "mode — dispatch_tile reads packed_ptrs and the destructor "
         "frees owned_packed_ptrs; an alias mismatch would either "
         "leak (destructor frees nothing) or use-after-free "
         "(dispatch_tile reads a freed pointer).";
}

TEST(CkPackBf16NoCache, NoLruInsertSecondPrepareDoesNotHitCache) {
  CK_SKIP_IF_NO_BF16_ISA();
  ::reset_grp_matmul_caches();
  WeightCacheOverride wc_off(0);

  ck_test::PrepCallCase c{};
  c.label = "no_lru_insert_distinct_packs";

  // Same shape + same caller-owned weight storage across both
  // prepares.  Under WEIGHT_CACHE=1 these would HIT the LRU on the
  // second call (covered by `SecondPrepareHitsCacheSamePackedPtr`
  // above); under WEIGHT_CACHE=0 the first call never inserts so the
  // second MUST allocate a fresh buffer.
  ck_test::PrepCallStorage storage;
  ck::CallContext kctx_a, kctx_b;
  ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success);
  ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_b),
            zendnnl::error_handling::status_t::success);

  ASSERT_NE(kctx_a.packed_ptrs[0], nullptr);
  ASSERT_NE(kctx_b.packed_ptrs[0], nullptr);
  EXPECT_NE(kctx_a.packed_ptrs[0], kctx_b.packed_ptrs[0])
      << "two prepares with the same (weight ptr, K, N, ldb, transB) "
         "produced the same packed pointer under WEIGHT_CACHE=0 — the "
         "LRU singleton is being consulted when it should be bypassed.";
  EXPECT_NE(kctx_a.owned_packed_ptrs[0],
            kctx_b.owned_packed_ptrs[0])
      << "owned_packed_ptrs must also be distinct — each prepare "
         "owns its own freshly-allocated arena.";
}

TEST(CkPackBf16NoCache, ResetReassignsPackedAliasAfterRepack) {
  CK_SKIP_IF_NO_BF16_ISA();
  ::reset_grp_matmul_caches();
  WeightCacheOverride wc_off(0);

  // Reuse a single `CallContext` across two prepares.  The second
  // prepare's implicit `reset()` frees the first prepare's owned
  // buffer (the contract that lets long-running pipelines reuse
  // one context per worker thread), then the per-expert pack loop
  // populates BOTH `packed_ptrs[i]` AND `owned_packed_ptrs[i]`
  // with the fresh alloc.
  //
  // Memory-recycling caveat:
  //   `release_owned_buffers()` calls `std::free` on the first
  //   call's buffer.  The libc allocator routinely returns the
  //   same address on a subsequent `std::aligned_alloc` of the
  //   same size+alignment from the same arena, so a pointer-NE
  //   assertion across the two calls would be flaky depending on
  //   surrounding test ordering and allocator state.  The real
  //   invariant this test pins is the post-reset alias: after the
  //   second prepare, `packed_ptrs[0]` must STILL equal
  //   `owned_packed_ptrs[0]` (and both must be non-null).  A bug
  //   that fails to re-assign `packed_ptrs[i]` in the pack loop
  //   would leave it pointing at the freed (or recycled) first-
  //   call buffer with NO owning slot — `dispatch_tile` would
  //   then read freed memory and the destructor would silently
  //   leak the second call's owning buffer.
  ck_test::PrepCallCase c{};
  c.label = "reset_realiases_packed_after_repack";

  ck_test::PrepCallStorage s1, s2;
  ck::CallContext kctx;
  ASSERT_EQ(ck_test::run_prepare(c, s1, kctx),
            zendnnl::error_handling::status_t::success);
  ASSERT_NE(kctx.owned_packed_ptrs[0], nullptr);

  ASSERT_EQ(ck_test::run_prepare(c, s2, kctx),
            zendnnl::error_handling::status_t::success);
  ASSERT_NE(kctx.owned_packed_ptrs[0], nullptr);
  EXPECT_EQ(static_cast<const void *>(kctx.packed_ptrs[0]),
            static_cast<const void *>(kctx.owned_packed_ptrs[0]))
      << "packed_ptrs[0] must alias owned_packed_ptrs[0] after the "
         "second prepare on a reused CallContext — a mismatch "
         "indicates the dispatcher's per-expert pack loop wrote to "
         "packed_ptrs without updating owned_packed_ptrs (or "
         "vice-versa), and dispatch_tile would read a stale or "
         "freed pointer.";
}

}  // namespace
