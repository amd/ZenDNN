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

  // ── [Negative] silu_and_mul / gelu_and_mul refused at the gate ─
  // The kernel cannot deinterleave the split-halves
  // [gate_cols | up_cols] layout in its per-tile epilogue, and the
  // dispatcher cannot enforce that a direct `prepare_for_call`
  // caller actually runs the post-activation pass over the wide
  // matmul output.  Production callers translate these activations
  // to `act = none` BEFORE invoking `prepare_for_call` and run
  // `group_matmul_moe_act_execute` themselves on the matmul output;
  // the gate refuses split-halves activations directly so a future
  // direct caller can't accidentally get only matmul output.
  // Iterate over (bias, dst) so the refusal is asserted across the
  // full dtype space — the gate fires before the dst/bias gates are
  // reached.
  for (auto act : {grp_matmul_gated_act_t::silu_and_mul,
                   grp_matmul_gated_act_t::gelu_and_mul}) {
    const char *act_name =
        (act == grp_matmul_gated_act_t::silu_and_mul) ? "silu" : "gelu";
    for (auto bias : {data_type_t::none, data_type_t::bf16,
                      data_type_t::f32}) {
      for (auto dst : {data_type_t::bf16, data_type_t::f32}) {
        auto c = baseline(
            std::string("neg_") + act_name
            + "_bias_" + ck_test::dt_name(bias)
            + "_dst_"  + ck_test::dt_name(dst),
            /*expect_success=*/false);
        c.act     = act;
        c.bias_dt = bias;
        c.dst_dt  = dst;
        cases.push_back(c);
      }
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
// [Property] silu_and_mul / gelu_and_mul are refused at the gate.
// `prepare_for_call` cannot enforce that a direct caller will run
// the post-activation pass on the wide matmul output, so the gate
// rejects split-halves activations and forces the caller to
// translate them to `act = none` first (which is what `flat_n_tile`
// does in production).  A regression that accidentally accepts
// these activations would silently leave a direct caller with only
// matmul output and no activation applied.
// ──────────────────────────────────────────────────────────────────
TEST(CkPrepareForCallProperties, SiluGeluRefusedAtGate) {
  CK_SKIP_IF_NO_BF16_ISA();
  for (auto act : {grp_matmul_gated_act_t::silu_and_mul,
                   grp_matmul_gated_act_t::gelu_and_mul}) {
    ck_test::PrepCallCase c{};
    c.act = act;
    c.label = "silu_gelu_refused_at_gate";
    ck_test::PrepCallStorage storage;
    ck::CallContext kctx;
    EXPECT_EQ(ck_test::run_prepare(c, storage, kctx), status_t::failure)
        << "act=" << static_cast<int>(act)
        << " — split-halves activations must be refused at the "
           "gate so direct callers cannot get matmul-only output "
           "without realising the post-pass is missing.";
    EXPECT_FALSE(kctx.enabled);
    // `kctx.variant` is populated by `resolve_variant` BEFORE the
    // activation gate fires, so on refusal it can already hold the
    // (bf16, bf16, bf16) row's `kBF16_BF16_BF16` value.  The
    // contract on refusal is `enabled = false`; the variant field
    // is undefined.
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
