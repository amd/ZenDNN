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

/// `prepare_for_call()` gating matrix.
///
/// Tests every gate in the dispatcher's pre-flight check, one at a
/// time, so a refusal points cleanly at one rejected condition.
///
/// Strategy:
///   1. Establish a baseline `PrepCallCase` (defaults — known to
///      succeed today).  Every negative case below toggles ONE field
///      from baseline.
///   2. Every gate gets a paired (positive / negative) row.  Failure
///      of the negative case implies the gate is missing; failure of
///      the positive case implies a regression on a supported tuple.
///
/// All cases skip on a non-AVX-512-BF16 host where the dispatcher
/// refuses every shape regardless of the per-gate logic — see
/// CK_SKIP_IF_NO_BF16_ISA.

#include "ck_test_helpers.hpp"

namespace {

namespace ck = ck_test::ck;
using ck_test::data_type_t;
using ck_test::grp_matmul_gated_act_t;
using ck_test::status_t;

// ──────────────────────────────────────────────────────────────────
// Single parameterised suite — every row tests one gate at a time.
// `expect_success` encodes the expected verdict so positive +
// negative cases share infrastructure.
// ──────────────────────────────────────────────────────────────────
class CkPrepareForCallTest
    : public ::testing::TestWithParam<ck_test::PrepCallCase> {};

TEST_P(CkPrepareForCallTest, MatchesExpectedVerdict) {
  CK_SKIP_IF_NO_BF16_ISA();

  const auto &c = GetParam();
  ck_test::PrepCallStorage storage;
  ck::CallContext kctx;
  const auto status = ck_test::run_prepare(c, storage, kctx);

  if (c.expect_success) {
    ASSERT_EQ(status, status_t::success)
        << "case '" << c.label << "' expected success but refused";
    EXPECT_TRUE(kctx.enabled);
    EXPECT_NE(kctx.variant, ck::KernelVariant::kUnsupported);
    EXPECT_GT(kctx.pack_nr, 0);
    EXPECT_GT(kctx.NV, 0);
    EXPECT_GT(kctx.max_mr, 0);
  } else {
    ASSERT_EQ(status, status_t::failure)
        << "case '" << c.label << "' expected refusal but accepted";
    EXPECT_FALSE(kctx.enabled);
  }
}

// ──────────────────────────────────────────────────────────────────
// Helpers to build cases.  Default constructor of `PrepCallCase`
// provides a known-good baseline; mutators flip one field.
// ──────────────────────────────────────────────────────────────────
inline ck_test::PrepCallCase baseline(std::string label,
                                       bool expect_success = true) {
  ck_test::PrepCallCase c{};
  c.label          = std::move(label);
  c.expect_success = expect_success;
  return c;
}

// ──────────────────────────────────────────────────────────────────
// Build the parameter set.  Lambda factories return cases by value.
// ──────────────────────────────────────────────────────────────────
static std::vector<ck_test::PrepCallCase> make_prepare_cases() {
  std::vector<ck_test::PrepCallCase> cases;

  // ── [Positive] Supported tuples — both supported dst variants ──
  cases.push_back(baseline("pos_bf16_bf16_bf16_act_none"));
  {
    auto c = baseline("pos_bf16_bf16_f32_act_none");
    c.dst_dt = data_type_t::f32;
    cases.push_back(c);
  }
  {
    auto c = baseline("pos_bf16_bf16_bf16_act_swiglu");
    c.act = grp_matmul_gated_act_t::swiglu_oai_mul;
    // Swiglu requires N % pack_nr == 0 and N/2 % 16 == 0; default
    // N=256 satisfies both.
    cases.push_back(c);
  }
  // Bias dtype variants on the supported tuple.
  for (auto bias : {data_type_t::none, data_type_t::bf16,
                    data_type_t::f32}) {
    auto c = baseline(std::string("pos_bias_") + ck_test::dt_name(bias));
    c.bias_dt = bias;
    cases.push_back(c);
  }

  // ── [Negative] Dtype rejections — every disallowed src/wei/dst ─
  for (auto wei : {data_type_t::f32, data_type_t::f16, data_type_t::s8,
                   data_type_t::u8}) {
    auto c = baseline(std::string("neg_wei_") + ck_test::dt_name(wei),
                      /*expect_success=*/false);
    c.wei_dt = wei;
    cases.push_back(c);
  }
  for (auto src : {data_type_t::f32, data_type_t::f16, data_type_t::s8,
                   data_type_t::u8}) {
    auto c = baseline(std::string("neg_src_") + ck_test::dt_name(src),
                      /*expect_success=*/false);
    c.src_dt = src;
    cases.push_back(c);
  }
  for (auto dst : {data_type_t::f16, data_type_t::s8, data_type_t::u8}) {
    auto c = baseline(std::string("neg_dst_") + ck_test::dt_name(dst),
                      /*expect_success=*/false);
    c.dst_dt = dst;
    cases.push_back(c);
  }

  // ── [Negative] Activation × dst constraint ─────────────────────
  // swiglu_oai_mul + FP32 dst is structurally invalid (the swiglu
  // store helper writes BF16 only) and is rejected at two distinct
  // gates inside prepare_for_call, depending on `act_dtype`:
  //
  //   1. With act_dtype = f32: the early gate
  //      `act != none && act_dtype != bf16` fires first, refusing
  //      with reason `unsupported_act_dtype`.
  //   2. With act_dtype = bf16: the act_dtype gate passes, and
  //      `fill_kfn_table` then fails because `select_ukernel(MR, NV,
  //      swiglu_oai_mul, kF32)` returns nullptr — refusal with
  //      reason `kfn_table_fill_failed`.
  //
  // Both rows must refuse so callers cannot accidentally route a
  // (swiglu, f32-dst) call into the kernel.  The fallback path
  // (AOCL DLP + a separate f32 swiglu pass) handles that combo
  // outside the custom kernel.
  {
    auto c = baseline("neg_swiglu_f32dst_actdt_f32",
                      /*expect_success=*/false);
    c.act    = grp_matmul_gated_act_t::swiglu_oai_mul;
    c.dst_dt = data_type_t::f32;
    c.act_dt = data_type_t::f32;
    cases.push_back(c);
  }
  {
    auto c = baseline("neg_swiglu_f32dst_actdt_bf16",
                      /*expect_success=*/false);
    c.act    = grp_matmul_gated_act_t::swiglu_oai_mul;
    c.dst_dt = data_type_t::f32;
    c.act_dt = data_type_t::bf16;
    cases.push_back(c);
  }

  // ── [Negative] Bias dtype outside {none, bf16, f32} ───────────
  for (auto bias : {data_type_t::f16, data_type_t::s8, data_type_t::u8}) {
    auto c = baseline(std::string("neg_bias_") + ck_test::dt_name(bias),
                      /*expect_success=*/false);
    c.bias_dt = bias;
    cases.push_back(c);
  }

  // ── [Negative] is_weights_const = false → CK refuses ─────────
  {
    auto c = baseline("neg_is_weights_const_false",
                      /*expect_success=*/false);
    c.is_wc = false;
    cases.push_back(c);
  }

  // ── [Negative] N % pack_nr != 0 ──────────────────────────────
  // Both pack_nr candidates {32, 64} fail when N is not divisible.
  // N=200 = 40*5 isn't divisible by 32 or 64, so plan_pack_nr returns 0
  // and prepare_for_call refuses.
  {
    auto c = baseline("neg_N_indivisible_by_pack_nr",
                      /*expect_success=*/false);
    c.N = 200;
    cases.push_back(c);
  }

  // ── [Negative] transA = true (kernel is non-transposed-A only) ─
  {
    auto c = baseline("neg_transA_true", /*expect_success=*/false);
    c.transA = true;
    cases.push_back(c);
  }

  // ── [Positive] Different valid shapes (sanity sweep) ─────────
  // Confirm the gate doesn't reject perfectly normal MoE shapes.
  struct ShapeRow { int M, K, N; const char *label; };
  for (const auto &s : {
           ShapeRow{1,    64,  256, "shape_M1"},
           ShapeRow{8,   128,  512, "shape_M8"},
           ShapeRow{32,  256, 1024, "shape_decode_med"},
           ShapeRow{128, 2880, 5760, "shape_K2880_N5760"},
           ShapeRow{4,   2048, 1536, "shape_K2048_N1536"},
       }) {
    auto c = baseline(std::string("pos_") + s.label);
    c.M = s.M;
    c.K = s.K;
    c.N = s.N;
    cases.push_back(c);
  }

  // ── [Positive] swiglu + bias dtypes (all three) ───────────────
  for (auto bias : {data_type_t::none, data_type_t::bf16,
                    data_type_t::f32}) {
    auto c = baseline(std::string("pos_swiglu_bias_")
                      + ck_test::dt_name(bias));
    c.act     = grp_matmul_gated_act_t::swiglu_oai_mul;
    c.bias_dt = bias;
    cases.push_back(c);
  }

  // ── [Positive] silu_and_mul / gelu_and_mul fused-CK path ──────
  // Prepack permutes canonical split-halves W13 into the
  // interleaved CK layout (silu and gelu share the SAME
  // permutation), and the in-register pair-store helpers
  // (`silu_and_mul_store_pair`, `gelu_and_mul_store_pair`) apply
  // the activation before the BF16 store.  Both gated-act +
  // BF16-dst tuples are valid here; FP32 dst is structurally
  // rejected (gated-act epilogue is BF16-only) and is asserted
  // below.  act_dtype must be BF16 (CK requires it for any fused
  // activation).
  for (auto act : {grp_matmul_gated_act_t::silu_and_mul,
                   grp_matmul_gated_act_t::gelu_and_mul}) {
    const char *act_name =
        (act == grp_matmul_gated_act_t::silu_and_mul) ? "silu" : "gelu";
    auto c = baseline(std::string("pos_") + act_name + "_no_bias_bf16dst");
    c.act     = act;
    c.bias_dt = data_type_t::none;
    cases.push_back(c);
  }

  // ── [Negative] silu_and_mul / gelu_and_mul + bias refused at gate ─
  // bias-into-init under the interleaved layout would have to read
  // [gate_bias | up_bias] in permuted order to match the prepack
  // permutation; both fused split-halves paths decline biased calls
  // through the same `split_halves_act_with_bias_not_fused` reason
  // string (planned follow-up).  Asserts for both supported bias
  // dtypes × both gated kinds = 4 cases.
  for (auto act : {grp_matmul_gated_act_t::silu_and_mul,
                   grp_matmul_gated_act_t::gelu_and_mul}) {
    const char *act_name =
        (act == grp_matmul_gated_act_t::silu_and_mul) ? "silu" : "gelu";
    for (auto bias : {data_type_t::bf16, data_type_t::f32}) {
      auto c = baseline(
          std::string("neg_") + act_name + "_bias_" + ck_test::dt_name(bias)
          + "_refused_at_gate",
          /*expect_success=*/false);
      c.act     = act;
      c.bias_dt = bias;
      cases.push_back(c);
    }
  }

  // ── [Negative] silu_and_mul / gelu_and_mul + FP32 dst refused ──
  // Symmetric with the swiglu_oai_mul rejection above — every
  // gated-act pair-pack store helper writes BF16 only.  Same
  // two-gate refusal pattern (act_dtype check vs
  // kfn_table_fill_failed depending on act_dt) for both gated kinds.
  for (auto act : {grp_matmul_gated_act_t::silu_and_mul,
                   grp_matmul_gated_act_t::gelu_and_mul}) {
    const char *act_name =
        (act == grp_matmul_gated_act_t::silu_and_mul) ? "silu" : "gelu";
    {
      auto c = baseline(std::string("neg_") + act_name + "_f32dst_actdt_f32",
                        /*expect_success=*/false);
      c.act    = act;
      c.dst_dt = data_type_t::f32;
      c.act_dt = data_type_t::f32;
      cases.push_back(c);
    }
    {
      auto c = baseline(std::string("neg_") + act_name + "_f32dst_actdt_bf16",
                        /*expect_success=*/false);
      c.act    = act;
      c.dst_dt = data_type_t::f32;
      c.act_dt = data_type_t::bf16;
      cases.push_back(c);
    }
  }

  return cases;
}

// gtest holds parameter sources for the lifetime of the test suite,
// and `::testing::ValuesIn(const Container&)` may store iterators
// into that container.  Passing a temporary `vector` would leave
// dangling iterators after the rvalue's destruction; wrap the
// builder in an immediately-invoked lambda whose function-local
// static gives the container static storage duration.
INSTANTIATE_TEST_SUITE_P(
    GatingMatrix, CkPrepareForCallTest,
    ::testing::ValuesIn(
        []() -> const std::vector<ck_test::PrepCallCase>& {
          static const std::vector<ck_test::PrepCallCase> kCases =
              make_prepare_cases();
          return kCases;
        }()),
    [](const ::testing::TestParamInfo<ck_test::PrepCallCase> &info) {
      return info.param.label;
    });

// ──────────────────────────────────────────────────────────────────
// [Property] silu_and_mul and gelu_and_mul are ACCEPTED (bias-free)
// and refused only when bias is present.
//
// Gate behaviour history:
//   * Pre-fuse: both silu_and_mul and gelu_and_mul were refused at
//     the gate; flat_n_tile translated them to `act = none` upstream
//     and ran the post-activation pass over the wide matmul output.
//   * silu fused: silu_and_mul accepted; prepack permutes canonical
//     split-halves W13 into the interleaved CK layout; in-register
//     `silu_and_mul_store_pair` applies silu before the BF16 store.
//     With-bias silu still refused (bias-into-init under the
//     interleaved layout is a follow-up).
//   * gelu fused (this change): gelu_and_mul accepted with the same
//     prepack permutation as silu; `gelu_and_mul_store_pair` applies
//     a `gelu_tanh` polynomial in registers (matches `gelu_erf`
//     within BF16 tolerance).  With-bias gelu also refused for the
//     same reason as silu.
//
// Regression coverage (six cases — each gated kind, three bias states):
//   1) silu_and_mul, no bias  → accept (was refused pre-fuse).
//   2) silu_and_mul, bf16 bias → refuse (forces caller to fall back).
//   3) silu_and_mul, f32 bias  → refuse.
//   4) gelu_and_mul, no bias  → accept (was refused pre-fuse).
//   5) gelu_and_mul, bf16 bias → refuse.
//   6) gelu_and_mul, f32 bias  → refuse.
// ──────────────────────────────────────────────────────────────────
TEST(CkPrepareForCallProperties, SiluGeluAcceptedNoBias) {
  CK_SKIP_IF_NO_BF16_ISA();

  struct Row {
    grp_matmul_gated_act_t act;
    ck::ActKind            expect_act_kind;
    const char            *act_name;
  };
  for (const auto &row : {
           Row{grp_matmul_gated_act_t::silu_and_mul,
               ck::ActKind::silu_and_mul, "silu"},
           Row{grp_matmul_gated_act_t::gelu_and_mul,
               ck::ActKind::gelu_and_mul, "gelu"},
       }) {
    // (a) no bias — ACCEPT.
    {
      ck_test::PrepCallCase c{};
      c.act     = row.act;
      c.bias_dt = data_type_t::none;
      c.label   = std::string(row.act_name) + "_no_bias_accepted_at_gate";
      ck_test::PrepCallStorage storage;
      ck::CallContext kctx;
      EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success)
          << row.act_name << "_and_mul (no bias) must now be accepted "
             "at the gate and route through the fused-CK in-register "
             "epilogue.";
      EXPECT_TRUE(kctx.enabled);
      EXPECT_EQ(kctx.act_kind, row.expect_act_kind)
          << "Accepted " << row.act_name << "_and_mul calls must map "
             "to the matching ActKind (the dispatcher's act_kind "
             "field) so dispatch_tile selects the right pair-store "
             "helper at the runtime branch.";
    }

    // (b) bf16 bias — REFUSE (bias-into-init under interleaved layout
    // is a planned follow-up; same restriction for silu and gelu).
    {
      ck_test::PrepCallCase c{};
      c.act     = row.act;
      c.bias_dt = data_type_t::bf16;
      c.label   = std::string(row.act_name) + "_with_bf16_bias_refused";
      ck_test::PrepCallStorage storage;
      ck::CallContext kctx;
      EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure)
          << row.act_name << "_and_mul + bf16 bias must be refused.";
      EXPECT_FALSE(kctx.enabled);
    }

    // (c) f32 bias — REFUSE (same reason).
    {
      ck_test::PrepCallCase c{};
      c.act     = row.act;
      c.bias_dt = data_type_t::f32;
      c.label   = std::string(row.act_name) + "_with_f32_bias_refused";
      ck_test::PrepCallStorage storage;
      ck::CallContext kctx;
      EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure)
          << row.act_name << "_and_mul + f32 bias must be refused.";
      EXPECT_FALSE(kctx.enabled);
    }
  }
}

// ──────────────────────────────────────────────────────────────────
// [Property] On a successful call, the resolved `kctx.variant` matches
// what `resolve_variant()` returns for the same (src, wei, dst).  This
// keeps the per-call gate's behaviour aligned with the routing-table
// gate — the same source of truth from the unit tests in
// test_resolve_variant.cpp.
// ──────────────────────────────────────────────────────────────────
TEST(CkPrepareForCallProperties, VariantFieldMatchesResolveVariant) {
  CK_SKIP_IF_NO_BF16_ISA();
  for (auto dst : {data_type_t::bf16, data_type_t::f32}) {
    auto c = baseline(std::string("variant_match_dst_")
                      + ck_test::dt_name(dst));
    c.dst_dt = dst;
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    ASSERT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::success);
    EXPECT_EQ(kctx.variant,
              ck::resolve_variant(c.src_dt, c.wei_dt, c.dst_dt));
  }
}

}  // namespace
