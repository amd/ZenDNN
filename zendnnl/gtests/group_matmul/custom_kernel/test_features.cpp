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

/// CK feature-integration tests — exercises the per-call CallContext
/// fields exposed to callers + the env-knob escape hatches that wrap
/// the kernel.
///
/// What's covered here:
///   * `kctx.act_kind` / `kctx.bias_kind` post-prepare values match
///     the inputs.  Locks down the resolved-act / bias-kind contract
///     callers rely on for per-tile branching.
///   * `kctx.subtile_cols_per_expert[]` populated when the env knob
///     `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT=1` is on,
///     zero-filled by default.
///   * `kctx.subtile_cols` (representative) is positive and a
///     multiple of `pack_nr` post-prepare.
///   * Multi-expert prepare populates `kctx.packed_ptrs` for every
///     active expert and leaves the inactive tail at nullptr.
///
/// These are CK-internal contracts — callers in
/// `group_matmul_n_tile.cpp` and `prepack_custom_kernel.cpp` rely on
/// them, but they're not exercised by the existing end-to-end suite.

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "ck_test_helpers.hpp"

namespace {

namespace ck = ck_test::ck;
using ck_test::bfloat16_t;
using ck_test::data_type_t;
using ck_test::grp_matmul_gated_act_t;
using ck_test::status_t;

// ──────────────────────────────────────────────────────────────────
// Resolved-fields contract: `kctx.act_kind` and `kctx.bias_kind` mirror
// the inputs to `prepare_for_call` per the documented mapping.
// ──────────────────────────────────────────────────────────────────
TEST(CkFeatures, ResolvedActKindMatches) {
  CK_SKIP_IF_NO_BF16_ISA();
  for (auto act : {grp_matmul_gated_act_t::none,
                   grp_matmul_gated_act_t::swiglu_oai_mul}) {
    ck_test::PrepCallCase c{};
    c.act = act;
    c.label = "act_resolved";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
    if (act == grp_matmul_gated_act_t::swiglu_oai_mul) {
      EXPECT_EQ(kctx.act_kind, ck::ActKind::swiglu_oai_mul);
    } else {
      EXPECT_EQ(kctx.act_kind, ck::ActKind::none);
    }
  }
}

TEST(CkFeatures, ResolvedBiasKindMatches) {
  CK_SKIP_IF_NO_BF16_ISA();
  struct Row { data_type_t dt; ck::BiasKind expected; };
  for (auto r : {Row{data_type_t::none, ck::BiasKind::none},
                 Row{data_type_t::bf16, ck::BiasKind::bf16},
                 Row{data_type_t::f32 , ck::BiasKind::fp32}}) {
    ck_test::PrepCallCase c{};
    c.bias_dt = r.dt;
    c.label = "bias_resolved";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.bias_kind, r.expected)
        << "bias_dt=" << ck_test::dt_name(r.dt);
  }
}

// ──────────────────────────────────────────────────────────────────
// `subtile_cols` is positive and a multiple of `pack_nr` after
// prepare — the per-tile path divides the per-thread N range into
// `subtile_cols`-wide chunks via integer arithmetic, so a
// non-multiple value would silently truncate.
// ──────────────────────────────────────────────────────────────────
TEST(CkFeatures, RepresentativeSubtileIsValid) {
  CK_SKIP_IF_NO_BF16_ISA();
  for (int N : {64, 128, 256, 512, 1536, 5760}) {
    ck_test::PrepCallCase c{};
    c.N = N;
    c.label = "subtile_valid";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
    EXPECT_GT(kctx.subtile_cols, 0)
        << "subtile_cols not populated; N=" << N;
    EXPECT_EQ(kctx.subtile_cols % kctx.pack_nr, 0)
        << "subtile_cols=" << kctx.subtile_cols
        << " not a multiple of pack_nr=" << kctx.pack_nr
        << " (N=" << N << ")";
  }
}

// ──────────────────────────────────────────────────────────────────
// `subtile_cols_per_expert` knob contract.  Two halves of the
// truth-table, each pinned via `CustomKernelSubtilePerExpertOverride`
// so the assertion is deterministic regardless of whatever env state
// the cached `static const` getter happened to snapshot earlier in
// the process.
//
// Both tests use THREE active experts so slot 0 is genuinely active
// (the previous single-expert test only asserted on slots [1..] —
// always zero by inactivity, regardless of the knob — and so passed
// trivially without exercising the contract).
// ──────────────────────────────────────────────────────────────────

// OFF: every active slot in `subtile_cols_per_expert` stays zero;
// `dispatch_tile()` reads `subtile_cols` (the m_max-sized global)
// for each expert.
TEST(CkFeatures, SubtilePerExpertOverride_OffLeavesActiveSlotsZero) {
  CK_SKIP_IF_NO_BF16_ISA();
  moe_test_utils::CustomKernelSubtilePerExpertOverride
      subtile_guard(/*value=*/0);

  constexpr int kNumActive = 3;
  constexpr int M = 8, K = 64, N = 256;
  std::vector<bfloat16_t> wei0(K * N, bfloat16_t(0.05f));
  std::vector<bfloat16_t> wei1(K * N, bfloat16_t(0.06f));
  std::vector<bfloat16_t> wei2(K * N, bfloat16_t(0.07f));
  std::vector<const void *> weight = {wei0.data(), wei1.data(),
                                      wei2.data()};
  std::vector<bool>  transA(kNumActive, false), transB(kNumActive, false);
  std::vector<bool>  is_wc(kNumActive, true);
  std::vector<int>   M_v(kNumActive, M), N_v(kNumActive, N),
                     K_v(kNumActive, K), ldb_v(kNumActive, N);
  std::vector<float> alpha_v(kNumActive, 1.0f),
                     beta_v(kNumActive, 0.0f);

  ck::CallContext kctx;
  ASSERT_EQ(ck::prepare_for_call(
                grp_matmul_gated_act_t::none,
                data_type_t::bf16, data_type_t::bf16, data_type_t::bf16,
                data_type_t::bf16, data_type_t::none,
                transA, transB, M_v, N_v, K_v, ldb_v, alpha_v, beta_v,
                weight, is_wc, kctx),
            status_t::success);

  EXPECT_GT(kctx.subtile_cols, 0)
      << "global subtile_cols not populated under OFF — `dispatch_tile`"
         " has no value to read for any expert";
  // Active slots [0, kNumActive) must be zero (per-expert path off).
  for (int i = 0; i < kNumActive; ++i) {
    EXPECT_EQ(kctx.subtile_cols_per_expert[i], 0)
        << "active slot " << i
        << " populated under override=OFF — should fall back to the"
           " global subtile_cols";
  }
}

// ON: every active slot is populated with a positive value sized
// from that expert's M; inactive tail stays zero.
TEST(CkFeatures, SubtilePerExpertOverride_OnPopulatesActiveSlots) {
  CK_SKIP_IF_NO_BF16_ISA();
  moe_test_utils::CustomKernelSubtilePerExpertOverride
      subtile_guard(/*value=*/1);

  constexpr int kNumActive = 3;
  constexpr int M = 8, K = 64, N = 256;
  std::vector<bfloat16_t> wei0(K * N, bfloat16_t(0.05f));
  std::vector<bfloat16_t> wei1(K * N, bfloat16_t(0.06f));
  std::vector<bfloat16_t> wei2(K * N, bfloat16_t(0.07f));
  std::vector<const void *> weight = {wei0.data(), wei1.data(),
                                      wei2.data()};
  std::vector<bool>  transA(kNumActive, false), transB(kNumActive, false);
  std::vector<bool>  is_wc(kNumActive, true);
  std::vector<int>   M_v(kNumActive, M), N_v(kNumActive, N),
                     K_v(kNumActive, K), ldb_v(kNumActive, N);
  std::vector<float> alpha_v(kNumActive, 1.0f),
                     beta_v(kNumActive, 0.0f);

  ck::CallContext kctx;
  ASSERT_EQ(ck::prepare_for_call(
                grp_matmul_gated_act_t::none,
                data_type_t::bf16, data_type_t::bf16, data_type_t::bf16,
                data_type_t::bf16, data_type_t::none,
                transA, transB, M_v, N_v, K_v, ldb_v, alpha_v, beta_v,
                weight, is_wc, kctx),
            status_t::success);

  // Active slots populated, must be a multiple of pack_nr (the per-
  // expert subtile arithmetic respects the same alignment as the
  // global `subtile_cols`).
  for (int i = 0; i < kNumActive; ++i) {
    EXPECT_GT(kctx.subtile_cols_per_expert[i], 0)
        << "active slot " << i
        << " not populated under override=ON";
    EXPECT_EQ(kctx.subtile_cols_per_expert[i] % kctx.pack_nr, 0)
        << "active slot " << i
        << " value=" << kctx.subtile_cols_per_expert[i]
        << " not a multiple of pack_nr=" << kctx.pack_nr;
  }
  // Inactive tail must stay zero — `dispatch_tile()` indexes by
  // expert and a stale non-zero would size the wrong subtile width.
  for (int i = kNumActive;
       i < static_cast<int>(kctx.subtile_cols_per_expert.size()); ++i) {
    EXPECT_EQ(kctx.subtile_cols_per_expert[i], 0)
        << "inactive slot " << i
        << " populated under override=ON";
  }
}

// ──────────────────────────────────────────────────────────────────
// Multi-expert prepare: every active expert's packed pointer is
// populated; inactive tail stays nullptr.  Important because the
// per-tile dispatch indexes by expert and a stale nullptr would
// crash, while a stale non-null would write to the wrong buffer.
// ──────────────────────────────────────────────────────────────────
TEST(CkFeatures, MultiExpertPreparePopulatesActiveOnly) {
  CK_SKIP_IF_NO_BF16_ISA();

  // 3 active experts on a uniform shape.  Build per-expert vectors
  // with the dispatcher's expected sizes.
  constexpr int kNumActive = 3;
  constexpr int M = 8, K = 64, N = 256;
  std::vector<bfloat16_t> wei0(K * N, bfloat16_t(0.05f));
  std::vector<bfloat16_t> wei1(K * N, bfloat16_t(0.06f));
  std::vector<bfloat16_t> wei2(K * N, bfloat16_t(0.07f));
  std::vector<const void *> weight = {wei0.data(), wei1.data(),
                                      wei2.data()};
  std::vector<bool>  transA(kNumActive, false), transB(kNumActive, false);
  std::vector<bool>  is_wc(kNumActive, true);
  std::vector<int>   M_v(kNumActive, M), N_v(kNumActive, N),
                     K_v(kNumActive, K), ldb_v(kNumActive, N);
  std::vector<float> alpha_v(kNumActive, 1.0f),
                     beta_v(kNumActive, 0.0f);

  ck::CallContext kctx;
  const auto status = ck::prepare_for_call(
      grp_matmul_gated_act_t::none, data_type_t::bf16, data_type_t::bf16,
      data_type_t::bf16, data_type_t::bf16, data_type_t::none,
      transA, transB, M_v, N_v, K_v, ldb_v, alpha_v, beta_v,
      weight, is_wc, kctx);
  ASSERT_EQ(status, status_t::success);

  // First `kNumActive` slots populated; rest stay nullptr.
  for (int i = 0; i < kNumActive; ++i) {
    EXPECT_NE(kctx.packed_ptrs[i], nullptr)
        << "active expert " << i << " has nullptr packed_ptr";
  }
  for (int i = kNumActive;
       i < static_cast<int>(kctx.packed_ptrs.size()); ++i) {
    EXPECT_EQ(kctx.packed_ptrs[i], nullptr)
        << "inactive slot " << i << " has non-null packed_ptr";
  }
}

}  // namespace
