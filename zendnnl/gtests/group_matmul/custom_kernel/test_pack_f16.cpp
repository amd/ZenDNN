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

/// FP16 CK pack module tests -- sibling of test_pack_bf16.cpp for the
/// native AVX-512-FP16 weight pack consumed by the FP16 microkernel.
///
/// The FP16 pack is a SEPARATE surface from the bf16 / int8 packs:
///   * its own LRU singleton keyed with `kCustomKernelF16Marker`
///     (disjoint from the bf16 / int8 caches),
///   * its own `get_or_pack_weight_f16` / `free_owned_packed_weight_f16`
///     / `clear_custom_kernel_pack_cache_f16` entry points,
///   * a DIFFERENT physical layout -- `packed[O/pack_nr][K][pack_nr]`
///     with NO K-pair interleave and no K-pad (the native FP16 FMA
///     consumes one K-element per lane), vs the bf16 path's
///     `[O/pack_nr][K_pair][pack_nr][2]` VNNI K-pair layout, and
///   * it populates `kctx.packed_ptrs_f16` / `kctx.owned_packed_ptrs_f16`
///     (NOT the bf16 `packed_ptrs` / `owned_packed_ptrs` arrays).
///
/// This file mirrors the bf16 pack suite's cache-warm / distinct-
/// pointer / no-cache / silu-gelu-interleave coverage on the f16 pack
/// path, reading the f16-specific CallContext arrays.
///
/// NOT duplicated here: the `plan_pack_nr` truth-table suites
/// (`CkPlanPackNrTest` / `CkPlanPackNrOverride64Test`).  `plan_pack_nr`
/// is a pure function of (K, N) shared by all three dtype families, so
/// the bf16 file's coverage applies verbatim -- re-running it under an
/// "f16" banner would assert identical behaviour with no added value.
///
/// All tests gate on `CK_SKIP_IF_NO_F16_ISA()`: on a host / toolchain
/// without native AVX-512-FP16 the f16 pack path is never reached
/// (prepare_for_call refuses at the ISA gate and the call would route
/// to AOCL DLP), so the pack-pointer assertions would be vacuous.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "ck_test_helpers.hpp"
#include "common/op_config.hpp"

namespace {

namespace ck = ck_test::ck;
namespace mt = moe_test_utils;
using mt::float16_t;
using zendnnl::common::data_type_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

// RAII guard for the library-wide weight-cache toggle (copy of the
// helper in test_pack_bf16.cpp -- each file keeps it file-local in its
// own anonymous namespace).  Save the current
// `matmul_config_t::get_weight_cache()` value at construction, set the
// requested value, restore at destruction.
class WeightCacheOverride {
public:
    explicit WeightCacheOverride(int32_t value)
        : prev_(zendnnl::common::matmul_config_t::instance()
                          .get_weight_cache()) {
        zendnnl::common::matmul_config_t::instance().set_weight_cache(value);
    }
    ~WeightCacheOverride() {
        zendnnl::common::matmul_config_t::instance().set_weight_cache(prev_);
    }
    WeightCacheOverride(const WeightCacheOverride &) = delete;
    WeightCacheOverride &operator=(const WeightCacheOverride &) = delete;

private:
    int32_t prev_;
};

// Build a PrepCallCase pinned to the FP16 family (src=wei=dst=f16,
// act_dtype=f16 so any fused-activation gate sees the right dtype).
inline ck_test::PrepCallCase f16_case(const char *label) {
    ck_test::PrepCallCase c {};
    c.src_dt = data_type_t::f16;
    c.wei_dt = data_type_t::f16;
    c.dst_dt = data_type_t::f16;
    c.act_dt = data_type_t::f16;
    c.label = label;
    return c;
}

// ------------------------------------------------------------------
// prepare_for_call-side: post-prepare `kctx.pack_nr` matches what
// `plan_pack_nr` reports for the same (K, N).  Identical contract to
// the bf16 sibling -- pack_nr selection is dtype-agnostic, but this
// pins that the f16 prepare path threads it through correctly.
// ------------------------------------------------------------------
TEST(CkPackF16, PrepareReportsPlanPackNr) {
    CK_SKIP_IF_NO_F16_ISA();
    mt::CustomKernelNROverride nr_guard(0);
    for (int N : {32, 64, 96, 128, 256, 512, 1024, 1536, 2880, 5760}) {
        ck_test::PrepCallCase c
                = f16_case((std::string("ppnr_N") + std::to_string(N)).c_str());
        c.N = N;
        ck_test::PrepCallStorage storage;
        ck::CallContext kctx;
        const auto status = ck_test::run_prepare(c, storage, kctx);
        ASSERT_EQ(status, zendnnl::error_handling::status_t::success)
                << "case=" << c.label;
        EXPECT_EQ(kctx.pack_nr, ck::plan_pack_nr(c.K, c.N))
                << "kctx.pack_nr deviates from plan_pack_nr -- case="
                << c.label;
    }
}

// ------------------------------------------------------------------
// Cache-warm symmetry: same weight pointer + (K, N, ldb) produces the
// same packed pointer across two prepares (the f16 LRU hits on the
// second call).  Reads `packed_ptrs_f16` -- the f16 pack populates the
// f16-specific array, NOT the bf16 `packed_ptrs`.
// ------------------------------------------------------------------
TEST(CkPackF16, SecondPrepareHitsCacheSamePackedPtr) {
    CK_SKIP_IF_NO_F16_ISA();
    // Reset the process-wide pack caches before this test -- the
    // "same weight ptr + shape -> same packed pointer" assertion only
    // holds when the f16 cache starts empty (heap-address reuse across
    // stack-local storage could otherwise produce a stale first-prepare
    // HIT; see the bf16 sibling's note).
    ::reset_grp_matmul_caches();

    ck_test::PrepCallCase c = f16_case("f16_cache_warm_symmetry");

    ck_test::PrepCallStorage storage;
    ck::CallContext kctx_a, kctx_b;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success);
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_b),
            zendnnl::error_handling::status_t::success);

    EXPECT_NE(kctx_a.packed_ptrs_f16[0], nullptr);
    EXPECT_EQ(kctx_a.packed_ptrs_f16[0], kctx_b.packed_ptrs_f16[0])
            << "f16 cache-warm symmetry broken: same (weight ptr, K, N, ldb) "
               "produced different packed pointers across calls -- the f16 "
               "LRU pack cache likely has a key inconsistency";
}

// ------------------------------------------------------------------
// Different weight pointer -> different packed pointer.  Ensures the
// f16 LRU is keyed on the source pointer (not just (K, N)).
// ------------------------------------------------------------------
TEST(CkPackF16, DistinctWeightPointersProduceDistinctPacks) {
    CK_SKIP_IF_NO_F16_ISA();
    ck_test::PrepCallCase c = f16_case("f16_distinct_weight_ptrs");

    ck_test::PrepCallStorage s1, s2;
    ck::CallContext kctx1, kctx2;
    ASSERT_EQ(ck_test::run_prepare(c, s1, kctx1),
            zendnnl::error_handling::status_t::success);
    ASSERT_EQ(ck_test::run_prepare(c, s2, kctx2),
            zendnnl::error_handling::status_t::success);

    EXPECT_NE(kctx1.packed_ptrs_f16[0], nullptr);
    EXPECT_NE(kctx2.packed_ptrs_f16[0], nullptr);
    EXPECT_NE(kctx1.packed_ptrs_f16[0], kctx2.packed_ptrs_f16[0])
            << "two distinct f16 weight buffers produced the same packed "
               "pointer -- the f16 LRU is not keyed on weight pointer";
}

// ------------------------------------------------------------------
// silu_and_mul / gelu_and_mul interleave layout -- bit-equality with
// swiglu_oai_mul (f16 pack).  Sibling of the bf16
// `SiluGeluInterleavedPackMatchesSwigluBytes`.  After the prepack
// permutes a canonical split-halves W13 `[gate_cols | up_cols]` into
// the f16 CK arena, the bytes MUST be bit-identical to what
// swiglu_oai_mul produces from an already-interleaved W13
// `[g0, u0, g1, u1, ...]`.  silu and gelu share the same permutation
// (only the kernel-side activation math differs).
//
// FP16 layout difference vs bf16: the f16 pack is the plain
// `[O/pack_nr][K][pack_nr]` slab (no VNNI K-pair doubling, no K-pad),
// so the per-expert packed size is exactly `K * N * sizeof(float16_t)`.
// ------------------------------------------------------------------
TEST(CkPackF16, SiluGeluInterleavedPackMatchesSwigluBytes) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();

    constexpr int kK = 64;
    constexpr int kN = 256;
    // static storage lets the lambda below use kI without capturing it: real
    // MSVC otherwise demands the capture (C3493), while Clang -Werror rejects
    // capturing a constexpr constant (-Wunused-lambda-capture).
    static constexpr int kI = kN / 2;

    auto val_gate = [](int k, int j) {
        return static_cast<float>(k * 31 + j) * 1.0e-3f;
    };
    auto val_up = [](int k, int j) {
        return static_cast<float>(k * 31 + j + kI) * 1.0e-3f;
    };

    std::vector<float16_t> w_interleaved(
            static_cast<size_t>(kK) * kN, float16_t(0.0f));
    // Distinct split-halves buffers -- distinct pointers force distinct
    // LRU keys so the gelu pack actually runs the prepack path instead
    // of HITting silu's earlier entry (silu / gelu share the
    // interleave=1 cache-key bit).
    std::vector<float16_t> w_split_silu(
            static_cast<size_t>(kK) * kN, float16_t(0.0f));
    std::vector<float16_t> w_split_gelu(
            static_cast<size_t>(kK) * kN, float16_t(0.0f));
    for (int k = 0; k < kK; ++k) {
        for (int j = 0; j < kI; ++j) {
            w_interleaved[k * kN + 2 * j + 0] = float16_t(val_gate(k, j));
            w_interleaved[k * kN + 2 * j + 1] = float16_t(val_up(k, j));
            w_split_silu[k * kN + j] = float16_t(val_gate(k, j));
            w_split_silu[k * kN + kI + j] = float16_t(val_up(k, j));
            w_split_gelu[k * kN + j] = float16_t(val_gate(k, j));
            w_split_gelu[k * kN + kI + j] = float16_t(val_up(k, j));
        }
    }

    std::vector<bool> transA_v {false};
    std::vector<bool> transB_v {false};
    std::vector<int> M_v {16};
    std::vector<int> N_v {kN};
    std::vector<int> K_v {kK};
    std::vector<int> ldb_v {kN};
    std::vector<float> alpha_v {1.0f};
    std::vector<float> beta_v {0.0f};
    std::vector<bool> is_wc_v {true};

    std::vector<const void *> wi_v {w_interleaved.data()};
    std::vector<const void *> ws_silu_v {w_split_silu.data()};
    std::vector<const void *> ws_gelu_v {w_split_gelu.data()};

    ck::CallContext kctx_swiglu, kctx_silu, kctx_gelu;
    ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::swiglu_oai_mul,
                      data_type_t::f16, data_type_t::f16, data_type_t::f16,
                      data_type_t::f16, data_type_t::none, transA_v, transB_v,
                      M_v, N_v, K_v, ldb_v, alpha_v, beta_v, wi_v, is_wc_v,
                      kctx_swiglu),
            zendnnl::error_handling::status_t::success);
    ASSERT_TRUE(kctx_swiglu.enabled);

    ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::silu_and_mul,
                      data_type_t::f16, data_type_t::f16, data_type_t::f16,
                      data_type_t::f16, data_type_t::none, transA_v, transB_v,
                      M_v, N_v, K_v, ldb_v, alpha_v, beta_v, ws_silu_v, is_wc_v,
                      kctx_silu),
            zendnnl::error_handling::status_t::success);
    ASSERT_TRUE(kctx_silu.enabled);

    ASSERT_EQ(ck::prepare_for_call(grp_matmul_gated_act_t::gelu_and_mul,
                      data_type_t::f16, data_type_t::f16, data_type_t::f16,
                      data_type_t::f16, data_type_t::none, transA_v, transB_v,
                      M_v, N_v, K_v, ldb_v, alpha_v, beta_v, ws_gelu_v, is_wc_v,
                      kctx_gelu),
            zendnnl::error_handling::status_t::success);
    ASSERT_TRUE(kctx_gelu.enabled);

    ASSERT_EQ(kctx_swiglu.pack_nr, kctx_silu.pack_nr);
    ASSERT_EQ(kctx_swiglu.pack_nr, kctx_gelu.pack_nr);

    const int pack_nr = kctx_swiglu.pack_nr;
    ASSERT_GT(pack_nr, 0);
    // FP16 plain `[O/pack_nr][K][pack_nr]` slab: one f16 element per
    // (o-block, k, col), no K-pair interleave/doubling.  The expression
    // `(kN / pack_nr) * kK * pack_nr` algebraically simplifies to
    // `kK * kN` (pack_nr divides kN exactly), i.e. the total is exactly
    // K * N elements = kK * kN * sizeof(float16_t) bytes.  Kept in the
    // unsimplified block form to mirror the packed slab's actual
    // `[O/pack_nr][K][pack_nr]` index structure.
    const size_t pack_bytes = static_cast<size_t>(kN / pack_nr) * kK * pack_nr
            * sizeof(float16_t);

    const auto *p_swiglu = kctx_swiglu.packed_ptrs_f16[0];
    const auto *p_silu = kctx_silu.packed_ptrs_f16[0];
    const auto *p_gelu = kctx_gelu.packed_ptrs_f16[0];
    ASSERT_NE(p_swiglu, nullptr);
    ASSERT_NE(p_silu, nullptr);
    ASSERT_NE(p_gelu, nullptr);

    EXPECT_EQ(0, std::memcmp(p_swiglu, p_silu, pack_bytes))
            << "f16 silu_and_mul pack bytes do not match swiglu_oai_mul pack "
               "bytes -- the silu in-register fused epilogue would "
               "deinterleave (g, u) from the wrong columns and produce "
               "silent-wrong activations.";
    EXPECT_EQ(0, std::memcmp(p_swiglu, p_gelu, pack_bytes))
            << "f16 gelu_and_mul pack bytes do not match swiglu_oai_mul pack "
               "bytes -- gelu shares the silu permutation; only the "
               "kernel-side activation math differs.";
    EXPECT_EQ(0, std::memcmp(p_silu, p_gelu, pack_bytes))
            << "f16 silu_and_mul and gelu_and_mul packs differ -- they should "
               "be byte-identical for the same logical weight (the prepack "
               "interleave is activation-agnostic).";
}

// ------------------------------------------------------------------
// ZENDNNL_MATMUL_WEIGHT_CACHE=0 (no-cache mode), f16 path.  Sibling of
// the bf16 `CkPackBf16NoCache` suite.  Under `set_weight_cache(0)` the
// runtime routes every per-expert f16 pack through
// `get_or_pack_weight_f16(..., disable_cache=true)`, allocating a
// fresh aligned buffer per call (owned by the CallContext, freed via
// `free_owned_packed_weight_f16`).  Pins three invariants on the
// f16-specific arrays:
//   1) CK still ENGAGES (no refuse -> no DLP fallback).
//   2) Two prepares with the SAME (weight ptr, K, N, ldb, transB)
//      produce DISTINCT packed pointers (cache bypassed).
//   3) `owned_packed_ptrs_f16[i]` aliases `packed_ptrs_f16[i]`.
// ------------------------------------------------------------------
TEST(CkPackF16NoCache, CkEngagesAndAllocatesCallerOwnedPacks) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();
    WeightCacheOverride wc_off(0);

    ck_test::PrepCallCase c
            = f16_case("f16_ck_engages_under_weight_cache_zero");
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx),
            zendnnl::error_handling::status_t::success)
            << "prepare_for_call must succeed under WEIGHT_CACHE=0 -- the "
               "no-cache mode switches the per-expert f16 pack to caller-owned "
               "buffers, NOT refuse CK entirely.";
    EXPECT_TRUE(kctx.enabled)
            << "kctx.enabled must remain true under WEIGHT_CACHE=0";
    EXPECT_NE(kctx.packed_ptrs_f16[0], nullptr);
    EXPECT_NE(kctx.owned_packed_ptrs_f16[0], nullptr)
            << "owned_packed_ptrs_f16[0] must be populated under "
               "WEIGHT_CACHE=0 so the CallContext destructor frees the "
               "caller-owned f16 buffer.";
    EXPECT_EQ(static_cast<const void *>(kctx.packed_ptrs_f16[0]),
            static_cast<const void *>(kctx.owned_packed_ptrs_f16[0]))
            << "packed_ptrs_f16[0] must alias owned_packed_ptrs_f16[0] in "
               "no-cache mode -- dispatch_tile reads packed_ptrs_f16 and the "
               "destructor frees owned_packed_ptrs_f16; an alias mismatch "
               "would either leak or use-after-free.";
}

TEST(CkPackF16NoCache, NoLruInsertSecondPrepareDoesNotHitCache) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();
    WeightCacheOverride wc_off(0);

    ck_test::PrepCallCase c = f16_case("f16_no_lru_insert_distinct_packs");
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx_a, kctx_b;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success);
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_b),
            zendnnl::error_handling::status_t::success);

    ASSERT_NE(kctx_a.packed_ptrs_f16[0], nullptr);
    ASSERT_NE(kctx_b.packed_ptrs_f16[0], nullptr);
    EXPECT_NE(kctx_a.packed_ptrs_f16[0], kctx_b.packed_ptrs_f16[0])
            << "two prepares with the same (weight ptr, K, N, ldb, transB) "
               "produced the same f16 packed pointer under WEIGHT_CACHE=0 -- "
               "the f16 LRU singleton is being consulted when it should be "
               "bypassed.";
    EXPECT_NE(kctx_a.owned_packed_ptrs_f16[0], kctx_b.owned_packed_ptrs_f16[0])
            << "owned_packed_ptrs_f16 must also be distinct -- each prepare "
               "owns its own freshly-allocated f16 arena.";
}

TEST(CkPackF16NoCache, ResetReassignsPackedAliasAfterRepack) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();
    WeightCacheOverride wc_off(0);

    // Reuse a single CallContext across two prepares.  The second
    // prepare's implicit reset() frees the first's owned f16 buffer,
    // then the per-expert pack loop must re-populate BOTH
    // packed_ptrs_f16[i] AND owned_packed_ptrs_f16[i] with the fresh
    // alloc.  (Pointer-NE across the two calls would be flaky -- the
    // allocator can recycle the freed address -- so the load-bearing
    // invariant is the post-reset alias, mirroring the bf16 sibling.)
    ck_test::PrepCallCase c
            = f16_case("f16_reset_realiases_packed_after_repack");
    ck_test::PrepCallStorage s1, s2;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, s1, kctx),
            zendnnl::error_handling::status_t::success);
    ASSERT_NE(kctx.owned_packed_ptrs_f16[0], nullptr);

    ASSERT_EQ(ck_test::run_prepare(c, s2, kctx),
            zendnnl::error_handling::status_t::success);
    ASSERT_NE(kctx.owned_packed_ptrs_f16[0], nullptr);
    EXPECT_EQ(static_cast<const void *>(kctx.packed_ptrs_f16[0]),
            static_cast<const void *>(kctx.owned_packed_ptrs_f16[0]))
            << "packed_ptrs_f16[0] must alias owned_packed_ptrs_f16[0] after "
               "the second prepare on a reused CallContext -- a mismatch "
               "indicates the f16 per-expert pack loop wrote to one array "
               "without updating the other, and dispatch_tile_f16 would read "
               "a stale or freed pointer.";
}

// ------------------------------------------------------------------
// ZENDNNL_MATMUL_WEIGHT_CACHE=2 (in-place mode) fallthrough, f16 path.
// The F16 pack family has NO in-place mode (unlike bf16 even-K, which
// writes the pack back into the caller's weight buffer and caches a
// nullptr sentinel under WC=2).  `get_or_pack_weight_f16` has no
// `in_place` parameter, so a WC=2 request must TRANSPARENTLY fall
// through to the out-of-place LRU.  This pins that contract (documented
// on `CallContext::owned_packed_ptrs_f16` / `get_or_pack_weight_f16`)
// so a future refactor that adds an f16 in-place path can't silently
// change it without updating this test.  Asserts:
//   1) CK still ENGAGES (no refuse -> no DLP fallback).
//   2) `owned_packed_ptrs_f16[0]` stays nullptr -- WC=2 is NOT the
//      cache-off (WC=0) mode, so no caller-owned buffer is allocated;
//      the pack lives in the LRU.
//   3) The packed pointer does NOT alias the caller's weight buffer --
//      the pack is out-of-place, not written in place into `weight`.
//   4) A second prepare on the same (weight ptr, K, N, ldb) HITs the
//      LRU and returns the SAME packed pointer (out-of-place cache is
//      live, not bypassed).
TEST(CkPackF16WeightCache2, FallsThroughToOutOfPlaceLru) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();
    WeightCacheOverride wc_inplace(2);

    ck_test::PrepCallCase c = f16_case("f16_wc2_falls_through_to_out_of_place");
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx_a;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success)
            << "prepare_for_call must succeed under WEIGHT_CACHE=2 -- the f16 "
               "family has no in-place mode, so WC=2 falls through to the "
               "out-of-place LRU, it does NOT refuse CK.";
    EXPECT_TRUE(kctx_a.enabled)
            << "kctx.enabled must remain true under WEIGHT_CACHE=2";
    ASSERT_NE(kctx_a.packed_ptrs_f16[0], nullptr);

    // The weight buffer `run_prepare` handed to the dispatcher.
    const void *weight_ptr
            = static_cast<const void *>(storage.wei_f16_storage.data());

    EXPECT_EQ(kctx_a.owned_packed_ptrs_f16[0], nullptr)
            << "owned_packed_ptrs_f16[0] must stay null under WEIGHT_CACHE=2 "
               "-- "
               "caller-owned buffers are the WC=0 (cache-off) path only; a "
               "non-null value would mean the f16 pack mistakenly took a "
               "caller-owned/in-place route.";
    EXPECT_NE(static_cast<const void *>(kctx_a.packed_ptrs_f16[0]), weight_ptr)
            << "packed_ptrs_f16[0] must NOT alias the caller's weight buffer "
               "under WEIGHT_CACHE=2 -- the f16 pack is out-of-place; aliasing "
               "the weight buffer would mean an (unsupported) in-place write-"
               "back sentinel leaked into the f16 path.";

    // Second prepare (same weight ptr + shape) must HIT the out-of-place
    // LRU and return the same packed pointer.
    ck::CallContext kctx_b;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_b),
            zendnnl::error_handling::status_t::success);
    ASSERT_EQ(static_cast<const void *>(storage.wei_f16_storage.data()),
            weight_ptr)
            << "test precondition: the reused storage kept the same weight "
               "buffer address across prepares";
    EXPECT_EQ(kctx_b.packed_ptrs_f16[0], kctx_a.packed_ptrs_f16[0])
            << "second prepare under WEIGHT_CACHE=2 produced a different f16 "
               "packed pointer -- the out-of-place LRU should have served the "
               "warmed entry for the same (weight ptr, K, N, ldb).";
    EXPECT_EQ(kctx_b.owned_packed_ptrs_f16[0], nullptr)
            << "owned_packed_ptrs_f16[0] must stay null on the WC=2 cache HIT "
               "as well.";
}

} // namespace
