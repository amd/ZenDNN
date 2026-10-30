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

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "ck_test_helpers.hpp"
#include "common/op_config.hpp"

namespace {

namespace ck = ck_test::ck;
namespace mt = moe_test_utils;
using mt::bfloat16_t;
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
    WeightCacheOverride wc_out_of_place(1);
    ::reset_grp_matmul_caches();
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
    WeightCacheOverride wc_out_of_place(1);
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
    WeightCacheOverride wc_out_of_place(1);
    ::reset_grp_matmul_caches();
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
    WeightCacheOverride wc_out_of_place(1);
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

// WC=2 writes a contiguous FP16 pack into the caller's buffer.  The cache
// records an in-place entry, so later prepares return the same caller pointer
// without giving CallContext ownership of it.
TEST(CkPackF16WeightCache2, PacksIntoCallerBuffer) {
    CK_SKIP_IF_NO_F16_ISA();
    ::reset_grp_matmul_caches();
    WeightCacheOverride wc_inplace(2);

    ck_test::PrepCallCase c = f16_case("f16_wc2_in_place");
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx_a;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx_a),
            zendnnl::error_handling::status_t::success)
            << "prepare_for_call must succeed under WEIGHT_CACHE=2";
    EXPECT_TRUE(kctx_a.enabled)
            << "kctx.enabled must remain true under WEIGHT_CACHE=2";
    ASSERT_NE(kctx_a.packed_ptrs_f16[0], nullptr);

    // The weight buffer `run_prepare` handed to the dispatcher.
    const void *weight_ptr
            = static_cast<const void *>(storage.wei_f16_storage.data());

    EXPECT_EQ(kctx_a.owned_packed_ptrs_f16[0], nullptr)
            << "the cache must not give CallContext ownership of caller memory";
    EXPECT_EQ(static_cast<const void *>(kctx_a.packed_ptrs_f16[0]), weight_ptr);
    EXPECT_TRUE(ck::weight_packed_in_place(weight_ptr));

    ::reset_grp_matmul_caches();
}

namespace f16_in_place {

using zendnnl::error_handling::status_t;

std::vector<float16_t> make_weight(size_t elems) {
    std::vector<float16_t> w(elems);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = float16_t(static_cast<float>(i % 251) * 0.03125f - 3.0f);
    }
    return w;
}

std::vector<char> reference_pack(const std::vector<float16_t> &raw, int K,
        int N, int ldb, int pack_nr, bool transB, bool interleave) {
    std::vector<char> out(ck::packed_weight_size_f16(K, N, pack_nr));
    EXPECT_EQ(ck::prepack_weight_into_f16(raw.data(), K, N, ldb, pack_nr,
                      transB, interleave, out.data()),
            status_t::success);
    return out;
}

const float16_t *pack(const std::vector<float16_t> &w, int K, int N, int ldb,
        int pack_nr, bool transB, bool interleave, bool disable_cache,
        bool want_in_place) {
    const float16_t *out = nullptr;
    EXPECT_EQ(ck::get_or_pack_weight_f16(w.data(), K, N, ldb, pack_nr, transB,
                      interleave, &out, /*was_hit_out=*/nullptr, disable_cache,
                      want_in_place),
            status_t::success);
    return out;
}

void expect_other_layouts_rebuilt(int K, int N, bool transB) {
    ::reset_grp_matmul_caches();
    const int ldb = transB ? K : N;
    std::vector<float16_t> w = make_weight(static_cast<size_t>(K) * N);
    const std::vector<float16_t> raw = w;
    const size_t bytes = static_cast<size_t>(K) * N * sizeof(float16_t);

    const float16_t *image = pack(w, K, N, ldb, /*pack_nr=*/32, transB,
            /*interleave=*/false, /*disable_cache=*/false,
            /*want_in_place=*/true);
    ASSERT_EQ(image, w.data());
    ASSERT_TRUE(ck::weight_packed_in_place(w.data()));
    EXPECT_EQ(0,
            std::memcmp(image,
                    reference_pack(raw, K, N, ldb, 32, transB, false).data(),
                    bytes));
    const std::vector<float16_t> image_bytes = w;

    for (const int pack_nr : {32, 64}) {
        for (const bool interleave : {false, true}) {
            if (pack_nr == 32 && !interleave) { continue; }
            SCOPED_TRACE("pack_nr=" + std::to_string(pack_nr)
                    + " interleave=" + std::to_string(interleave));
            const std::vector<char> ref = reference_pack(
                    raw, K, N, ldb, pack_nr, transB, interleave);

            const float16_t *cached = pack(w, K, N, ldb, pack_nr, transB,
                    interleave, /*disable_cache=*/false,
                    /*want_in_place=*/true);
            ASSERT_NE(cached, w.data()) << "an image was rewritten twice";
            EXPECT_EQ(0, std::memcmp(cached, ref.data(), bytes));

            const float16_t *owned = pack(w, K, N, ldb, pack_nr, transB,
                    interleave, /*disable_cache=*/true,
                    /*want_in_place=*/false);
            ASSERT_NE(owned, nullptr);
            EXPECT_EQ(0, std::memcmp(owned, ref.data(), bytes));
            ck::free_owned_packed_weight_f16(owned);

            std::vector<char> into(ck::packed_weight_size_f16(K, N, pack_nr));
            ASSERT_EQ(ck::prepack_weight_into_f16(w.data(), K, N, ldb, pack_nr,
                              transB, interleave, into.data()),
                    status_t::success);
            EXPECT_EQ(0, std::memcmp(into.data(), ref.data(), bytes));
        }
    }
    EXPECT_EQ(0,
            std::memcmp(w.data(), image_bytes.data(),
                    image_bytes.size() * sizeof(float16_t)))
            << "the image changed after it was written";
    EXPECT_EQ(pack(w, K, N, ldb, 32, transB, false, false, true), w.data());

    ::reset_grp_matmul_caches();
    EXPECT_FALSE(ck::weight_packed_in_place(w.data()));
}

} // namespace f16_in_place

TEST(CkPackF16InPlace, OtherLayoutsArePackedFromTheRebuiltWeight) {
    f16_in_place::expect_other_layouts_rebuilt(
            /*K=*/63, /*N=*/128, /*transB=*/false);
}

TEST(CkPackF16InPlace, OtherLayoutsArePackedFromTheRebuiltTransposedWeight) {
    f16_in_place::expect_other_layouts_rebuilt(
            /*K=*/63, /*N=*/128, /*transB=*/true);
}

TEST(CkPackF16InPlace, OddKStillPacksInPlace) {
    ::reset_grp_matmul_caches();
    constexpr int K = 63, N = 128;
    std::vector<float16_t> w
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    const float16_t *out = f16_in_place::pack(w, K, N, N, 32, false, false,
            /*disable_cache=*/false, /*want_in_place=*/true);
    EXPECT_EQ(out, w.data());
    EXPECT_TRUE(ck::weight_packed_in_place(w.data()));
    ::reset_grp_matmul_caches();
}

TEST(CkPackF16InPlace, OutOfPlaceEntryIsNeverRewrittenInPlace) {
    ::reset_grp_matmul_caches();
    constexpr int K = 64, N = 128;
    std::vector<float16_t> w
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    const std::vector<float16_t> raw = w;

    const float16_t *oop = f16_in_place::pack(w, K, N, N, 32, false, false,
            /*disable_cache=*/false, /*want_in_place=*/false);
    ASSERT_NE(oop, nullptr);
    ASSERT_NE(oop, w.data());
    const float16_t *wc2 = f16_in_place::pack(w, K, N, N, 32, false, false,
            /*disable_cache=*/false, /*want_in_place=*/true);
    EXPECT_EQ(wc2, oop);
    EXPECT_FALSE(ck::weight_packed_in_place(w.data()));
    EXPECT_EQ(0,
            std::memcmp(w.data(), raw.data(), raw.size() * sizeof(float16_t)));
    ::reset_grp_matmul_caches();
}

TEST(CkPackF16InPlace, ClearDoesNotLeaveAStaleAddressVerdict) {
    ::reset_grp_matmul_caches();
    constexpr int K = 64, N = 128;
    std::vector<float16_t> w
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    ASSERT_EQ(f16_in_place::pack(w, K, N, N, 32, false, false, false, true),
            w.data());
    ASSERT_TRUE(ck::weight_packed_in_place(w.data()));

    ck::clear_custom_kernel_pack_cache_f16();
    EXPECT_FALSE(ck::weight_packed_in_place(w.data()));
    // Simulate allocator address reuse by restoring a different raw image at
    // the exact same address before the next model/layer is packed.
    const auto replacement
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    std::copy(replacement.rbegin(), replacement.rend(), w.begin());
    const std::vector<float16_t> new_raw = w;
    const auto ref
            = f16_in_place::reference_pack(new_raw, K, N, N, 32, false, false);
    ASSERT_EQ(f16_in_place::pack(w, K, N, N, 32, false, false, false, true),
            w.data());
    EXPECT_EQ(0,
            std::memcmp(w.data(), ref.data(),
                    static_cast<size_t>(K) * N * sizeof(float16_t)));
    EXPECT_TRUE(ck::weight_packed_in_place(w.data()));
    ::reset_grp_matmul_caches();
}

TEST(CkPackF16InPlace, StridedAndInteriorViewsAreRefusedSafely) {
    ::reset_grp_matmul_caches();
    using f16_in_place::status_t;
    constexpr int K = 64, N = 128;

    std::vector<float16_t> padded
            = f16_in_place::make_weight(static_cast<size_t>(K) * (N + 8));
    const std::vector<float16_t> raw = padded;
    const float16_t *strided = f16_in_place::pack(padded, K, N, N + 8, 32,
            false, false, /*disable_cache=*/false, /*want_in_place=*/true);
    ASSERT_NE(strided, nullptr);
    EXPECT_NE(strided, padded.data());
    EXPECT_EQ(0,
            std::memcmp(
                    padded.data(), raw.data(), raw.size() * sizeof(float16_t)));

    std::vector<float16_t> w
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    ASSERT_EQ(f16_in_place::pack(w, K, N, N, 32, false, false, false, true),
            w.data());
    const float16_t *oversized = nullptr;
    EXPECT_EQ(ck::get_or_pack_weight_f16(w.data(), K, N, N + 32, 32, false,
                      false, &oversized, nullptr, false, true),
            status_t::failure);
    const float16_t *rows = w.data() + static_cast<size_t>(K / 2) * N;
    EXPECT_TRUE(ck::weight_packed_in_place(rows));
    const float16_t *out = nullptr;
    EXPECT_EQ(ck::get_or_pack_weight_f16(rows, K / 2, N, N, 32, false, false,
                      &out, nullptr, false, true),
            status_t::failure);
    ::reset_grp_matmul_caches();
}

TEST(CkPackF16InPlace, RegistryKeepsFamiliesSeparate) {
    ::reset_grp_matmul_caches();
    using f16_in_place::status_t;
    constexpr int K = 64, N = 128;

    std::vector<bfloat16_t> bf16(static_cast<size_t>(K) * N, bfloat16_t(0.5f));
    const bfloat16_t *bf16_out = nullptr;
    ASSERT_EQ(ck::get_or_pack_weight_bf16(bf16.data(), K, N, N, 32, false,
                      false, &bf16_out, nullptr, false, true),
            status_t::success);
    ASSERT_EQ(bf16_out, bf16.data());

    const float16_t *wrong_family = nullptr;
    EXPECT_EQ(ck::get_or_pack_weight_f16(
                      reinterpret_cast<const float16_t *>(bf16.data()), K, N, N,
                      32, false, false, &wrong_family, nullptr, false, true),
            status_t::failure);

    std::vector<float16_t> f16
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    ASSERT_EQ(f16_in_place::pack(f16, K, N, N, 32, false, false, false, true),
            f16.data());
    ck::clear_custom_kernel_pack_cache_f16();
    EXPECT_FALSE(ck::weight_packed_in_place(f16.data()));
    EXPECT_TRUE(ck::weight_packed_in_place(bf16.data()));
    ck::clear_custom_kernel_pack_cache();
    EXPECT_FALSE(ck::weight_packed_in_place(bf16.data()));
}

TEST(CkPackF16InPlace, ConcurrentFirstPackPublishesOneImage) {
    ::reset_grp_matmul_caches();
    using f16_in_place::status_t;
    constexpr int K = 63, N = 128, Calls = 16;
    std::vector<float16_t> w
            = f16_in_place::make_weight(static_cast<size_t>(K) * N);
    const std::vector<float16_t> raw = w;
    std::vector<const float16_t *> out(Calls, nullptr);
    std::vector<status_t> status(Calls, status_t::failure);

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < Calls; ++i) {
        status[i] = ck::get_or_pack_weight_f16(w.data(), K, N, N, 32, false,
                false, &out[i], nullptr, false, true);
    }

    for (int i = 0; i < Calls; ++i) {
        EXPECT_EQ(status[i], status_t::success) << "call " << i;
        EXPECT_EQ(out[i], w.data()) << "call " << i;
    }
    const auto ref
            = f16_in_place::reference_pack(raw, K, N, N, 32, false, false);
    EXPECT_EQ(0,
            std::memcmp(w.data(), ref.data(),
                    static_cast<size_t>(K) * N * sizeof(float16_t)));
    EXPECT_TRUE(ck::weight_packed_in_place(w.data()));
    ::reset_grp_matmul_caches();
}

} // namespace
