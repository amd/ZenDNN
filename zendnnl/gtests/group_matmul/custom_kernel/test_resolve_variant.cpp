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

/// `resolve_variant()` — single source of truth for the
/// "(src, wei, dst) → KernelVariant" routing table.
///
/// Why test this directly:
///   * `prepare_for_call` consumes its result and refuses on
///     `kUnsupported`; the per-call gate's correctness depends on
///     this table being right.
///   * Pinning the existing supported rows catches any accidental
///     regression on them when other dtype combinations are added.
///   * `dispatch_supported()` is irrelevant — this is a pure switch
///     over POD enums, runs identically on every host.
///
/// Coverage strategy: enumerate every value declared in
/// `data_type_t` (see `common/data_types.hpp`) on each of
/// (src, wei, dst) and assert the resulting variant.  At 13
/// dtype values that's 13³ = 2197 combinations — `resolve_variant`
/// is a pure switch with no I/O so the sweep is cheap and proves
/// every dtype outside the served set lands on `kUnsupported`,
/// including the rarely-used integer widths and the f16 row.

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "ck_test_helpers.hpp"

namespace {

namespace ck = ck_test::ck;
using ck_test::data_type_t;

// ──────────────────────────────────────────────────────────────────
// Hand-rolled positive-list — every (src, wei, dst) tuple that the
// kernel currently admits, with the variant it must resolve to.
// Negative cases are the complement on the full data_type_t³ sweep.
// ──────────────────────────────────────────────────────────────────
struct PositiveRow {
  data_type_t        src, wei, dst;
  ck::KernelVariant  expected;
};

constexpr PositiveRow kPositiveTable[] = {
    {data_type_t::bf16, data_type_t::bf16, data_type_t::bf16,
     ck::KernelVariant::kBF16_BF16_BF16},
    {data_type_t::bf16, data_type_t::bf16, data_type_t::f32,
     ck::KernelVariant::kBF16_BF16_F32},
};

// Every value declared in `data_type_t` (see `common/data_types.hpp`).
// Keep this in sync with the enum so the negative sweep cannot miss
// a newly added dtype quietly admitted by the dispatcher.
constexpr data_type_t kAllDtypes[] = {
    data_type_t::none,
    data_type_t::f32,
    data_type_t::f16,
    data_type_t::bf16,
    data_type_t::s32,
    data_type_t::s64,
    data_type_t::s16,
    data_type_t::s8,
    data_type_t::s4,
    data_type_t::u32,
    data_type_t::u16,
    data_type_t::u8,
    data_type_t::u4,
};

// ──────────────────────────────────────────────────────────────────
// [Positive] Every supported tuple resolves to its expected variant.
// ──────────────────────────────────────────────────────────────────
class CkResolveVariantPositive
    : public ::testing::TestWithParam<PositiveRow> {};

TEST_P(CkResolveVariantPositive, MatchesExpectedVariant) {
  const auto &p = GetParam();
  EXPECT_EQ(ck::resolve_variant(p.src, p.wei, p.dst), p.expected)
      << "src="  << ck_test::dt_name(p.src)
      << " wei=" << ck_test::dt_name(p.wei)
      << " dst=" << ck_test::dt_name(p.dst);
}

INSTANTIATE_TEST_SUITE_P(
    AllSupportedTuples, CkResolveVariantPositive,
    ::testing::ValuesIn(std::begin(kPositiveTable),
                        std::end(kPositiveTable)),
    [](const ::testing::TestParamInfo<PositiveRow> &info) {
      std::string s;
      s += ck_test::dt_name(info.param.src);
      s += "_";
      s += ck_test::dt_name(info.param.wei);
      s += "_";
      s += ck_test::dt_name(info.param.dst);
      return s;
    });

// ──────────────────────────────────────────────────────────────────
// [Negative] Every tuple NOT in the positive list resolves to
// `kUnsupported`.  Iterates 13³ = 2197 combinations (every value
// declared in `data_type_t`), skipping the supported rows, so the
// suite size stays in sync with the table above and an accidentally
// admitted dtype anywhere in the enum gets caught.
// ──────────────────────────────────────────────────────────────────
struct NegativeRow {
  data_type_t src, wei, dst;
};

inline std::vector<NegativeRow> build_negative_rows() {
  std::vector<NegativeRow> rows;
  // 13 dtypes ³ - small positive table; reserving a generous size
  // avoids reallocations during the build loop.
  rows.reserve(13 * 13 * 13);
  for (auto src : kAllDtypes) {
    for (auto wei : kAllDtypes) {
      for (auto dst : kAllDtypes) {
        bool is_positive = false;
        for (const auto &p : kPositiveTable) {
          if (p.src == src && p.wei == wei && p.dst == dst) {
            is_positive = true;
            break;
          }
        }
        if (!is_positive) {
          rows.push_back({src, wei, dst});
        }
      }
    }
  }
  return rows;
}

class CkResolveVariantNegative
    : public ::testing::TestWithParam<NegativeRow> {};

TEST_P(CkResolveVariantNegative, ResolvesToUnsupported) {
  const auto &p = GetParam();
  EXPECT_EQ(ck::resolve_variant(p.src, p.wei, p.dst),
            ck::KernelVariant::kUnsupported)
      << "src="  << ck_test::dt_name(p.src)
      << " wei=" << ck_test::dt_name(p.wei)
      << " dst=" << ck_test::dt_name(p.dst)
      << " — if you just landed an int8 (or other) variant, add the "
         "tuple to the kPositiveTable above so it is asserted in the "
         "Positive suite instead.";
}

// gtest holds parameter sources for the lifetime of the test suite,
// and `::testing::ValuesIn(const Container&)` may store iterators
// into that container.  Passing a temporary `vector` would leave
// dangling iterators after the rvalue's destruction; wrap the
// builder in an immediately-invoked lambda whose function-local
// static gives the container static storage duration.
INSTANTIATE_TEST_SUITE_P(
    AllRejectedTuples, CkResolveVariantNegative,
    ::testing::ValuesIn([]() -> const std::vector<NegativeRow>& {
      static const std::vector<NegativeRow> kRows = build_negative_rows();
      return kRows;
    }()),
    [](const ::testing::TestParamInfo<NegativeRow> &info) {
      std::string s;
      s += ck_test::dt_name(info.param.src);
      s += "_";
      s += ck_test::dt_name(info.param.wei);
      s += "_";
      s += ck_test::dt_name(info.param.dst);
      return s;
    });

// ──────────────────────────────────────────────────────────────────
// [Properties] Hand-coded invariants the table must satisfy.  These
// catch silly regressions (e.g., someone returning `kBF16_BF16_BF16`
// for a tuple that should have stayed `kUnsupported`).
// ──────────────────────────────────────────────────────────────────
TEST(CkResolveVariantProperties, NoneOnAnyDtypeIsUnsupported) {
  // `data_type_t::none` is a sentinel — never a real input; should
  // never match any supported variant.
  for (auto dt1 : kAllDtypes) {
    for (auto dt2 : kAllDtypes) {
      EXPECT_EQ(
          ck::resolve_variant(data_type_t::none, dt1, dt2),
          ck::KernelVariant::kUnsupported);
      EXPECT_EQ(
          ck::resolve_variant(dt1, data_type_t::none, dt2),
          ck::KernelVariant::kUnsupported);
      EXPECT_EQ(
          ck::resolve_variant(dt1, dt2, data_type_t::none),
          ck::KernelVariant::kUnsupported);
    }
  }
}

TEST(CkResolveVariantProperties, F16IsAlwaysUnsupportedToday) {
  // F16 is not supported in any (src, wei, dst) slot — even though
  // the dispatcher's `dt_name` recognises it for logging.
  for (auto dt1 : kAllDtypes) {
    for (auto dt2 : kAllDtypes) {
      EXPECT_EQ(ck::resolve_variant(data_type_t::f16, dt1, dt2),
                ck::KernelVariant::kUnsupported);
      EXPECT_EQ(ck::resolve_variant(dt1, data_type_t::f16, dt2),
                ck::KernelVariant::kUnsupported);
      EXPECT_EQ(ck::resolve_variant(dt1, dt2, data_type_t::f16),
                ck::KernelVariant::kUnsupported);
    }
  }
}

TEST(CkResolveVariantProperties, U8IsAlwaysUnsupportedToday) {
  // U8 has no instantiated variant on the 3-arg (BF16-only) overload.
  // The 5-arg `dynamic_quant=true, compute=u8` form is the DQ-INT8
  // ASYM path; that one is tested separately in
  // `CkResolveVariantInt8.AcceptsAsymmetric` below.
  for (auto dt1 : kAllDtypes) {
    for (auto dt2 : kAllDtypes) {
      EXPECT_EQ(ck::resolve_variant(data_type_t::u8, dt1, dt2),
                ck::KernelVariant::kUnsupported);
      EXPECT_EQ(ck::resolve_variant(dt1, data_type_t::u8, dt2),
                ck::KernelVariant::kUnsupported);
      EXPECT_EQ(ck::resolve_variant(dt1, dt2, data_type_t::u8),
                ck::KernelVariant::kUnsupported);
    }
  }
}

// ──────────────────────────────────────────────────────────────────
// DQ-INT8 truth table — the 5-arg `resolve_variant` overload that
// the int8 dispatcher consults.  Two positive rows + a sweep that
// asserts everything else (including `dynamic_quant=true` with a
// non-s8 wei or non-bf16 src/dst) resolves to `kUnsupported`.
// ──────────────────────────────────────────────────────────────────
TEST(CkResolveVariantInt8, AcceptsSymmetric) {
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::bf16,
                                /*dynamic_quant=*/true,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kS8_S8_BF16_SYM);
}

TEST(CkResolveVariantInt8, AcceptsAsymmetric) {
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::bf16,
                                /*dynamic_quant=*/true,
                                /*compute_dtype=*/data_type_t::u8),
            ck::KernelVariant::kU8_S8_BF16_ASYM);
}

TEST(CkResolveVariantInt8, AcceptsF32Dst) {
  // FP32 dst is a served int8 variant (ukernel_f32dst): src=bf16,
  // wei=s8, dst=f32, dynamic_quant=true, compute=s8/u8.
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::f32, /*dynamic_quant=*/true,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kS8_S8_F32_SYM);
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::f32, /*dynamic_quant=*/true,
                                /*compute_dtype=*/data_type_t::u8),
            ck::KernelVariant::kU8_S8_F32_ASYM);
}

TEST(CkResolveVariantInt8, AcceptsGroupedPreQuantS8SrcSym) {
  // group_dynamic_quant pre-pass form: the src is ALREADY s8 and the
  // dynamic_quant flag has been CLEARED.  resolve_variant accepts
  // src==s8 directly (mirror of dispatch.cpp's grouped-s8 acceptance),
  // independent of the dynamic_quant flag.  compute=s8 -> symmetric.
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::bf16, /*dynamic_quant=*/false,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kS8_S8_BF16_SYM);
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::f32, /*dynamic_quant=*/false,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kS8_S8_F32_SYM);
  // Also accepted when dynamic_quant happens to still be true (the
  // discriminator is src==s8, not the flag).
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::bf16, /*dynamic_quant=*/true,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kS8_S8_BF16_SYM);
}

TEST(CkResolveVariantInt8, AcceptsGroupedPreQuantS8SrcAsym) {
  // compute=u8 -> asymmetric, for both bf16 and f32 dst.
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::bf16, /*dynamic_quant=*/false,
                                /*compute_dtype=*/data_type_t::u8),
            ck::KernelVariant::kU8_S8_BF16_ASYM);
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::f32, /*dynamic_quant=*/false,
                                /*compute_dtype=*/data_type_t::u8),
            ck::KernelVariant::kU8_S8_F32_ASYM);
}

TEST(CkResolveVariantInt8, GroupedPreQuantS8RequiresValidComputeAndShape) {
  // s8 src still needs wei=s8, dst in {bf16,f32}, compute in {s8,u8}.
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::bf16, false, data_type_t::none),
            ck::KernelVariant::kUnsupported)
      << "compute=none (no DQ-INT8 contract) must reject";
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::bf16,
                                data_type_t::bf16, false, data_type_t::s8),
            ck::KernelVariant::kUnsupported)
      << "wei!=s8 must reject";
  EXPECT_EQ(ck::resolve_variant(data_type_t::s8, data_type_t::s8,
                                data_type_t::s32, false, data_type_t::s8),
            ck::KernelVariant::kUnsupported)
      << "dst outside {bf16,f32} must reject";
}

TEST(CkResolveVariantInt8, RejectsDynamicQuantWithoutInt8WeiPathway) {
  // dynamic_quant=true is the trigger for the int8 path, but the
  // routing still requires src=bf16, wei=s8, dst ∈ {bf16, f32}; any
  // deviation must reject.
  EXPECT_EQ(ck::resolve_variant(data_type_t::f32, data_type_t::s8,
                                data_type_t::bf16, true,
                                data_type_t::s8),
            ck::KernelVariant::kUnsupported)
      << "non-bf16 src must not resolve to the int8 path";
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::bf16,
                                data_type_t::bf16, true,
                                data_type_t::s8),
            ck::KernelVariant::kUnsupported)
      << "non-s8 wei must not resolve to the int8 path";
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::s32, true,
                                data_type_t::s8),
            ck::KernelVariant::kUnsupported)
      << "dst outside {bf16, f32} must not resolve to the int8 path";
}

TEST(CkResolveVariantInt8, RejectsUnknownComputeDtype) {
  // compute_dtype must be s8 or u8; anything else (e.g. bf16, f32,
  // s32) is a contract violation and must resolve to kUnsupported.
  for (auto bad : {data_type_t::f32, data_type_t::bf16,
                   data_type_t::s32, data_type_t::s4,
                   data_type_t::s16}) {
    EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                  data_type_t::bf16, true, bad),
              ck::KernelVariant::kUnsupported)
        << "compute_dtype=" << ck_test::dt_name(bad)
        << " must not resolve to the int8 path";
  }
}

TEST(CkResolveVariantInt8, DynamicQuantFlagIsRequired) {
  // (bf16, s8, bf16) without `dynamic_quant=true` must NOT match
  // the int8 path — that combination is the static-quant case
  // which N-tile / CK does not handle, and `resolve_variant` should
  // refuse cleanly so the call falls back to AOCL DLP.
  EXPECT_EQ(ck::resolve_variant(data_type_t::bf16, data_type_t::s8,
                                data_type_t::bf16,
                                /*dynamic_quant=*/false,
                                /*compute_dtype=*/data_type_t::s8),
            ck::KernelVariant::kUnsupported);
}

TEST(CkResolveVariantProperties, NoExceptOnEverything) {
  // The contract on `resolve_variant` is `noexcept`.  Smoke-prove it
  // by sweeping the full data_type_t³ enumeration — gtest would
  // catch any accidental throw in the call below.
  for (auto src : kAllDtypes)
    for (auto wei : kAllDtypes)
      for (auto dst : kAllDtypes)
        (void)ck::resolve_variant(src, wei, dst);
  static_assert(noexcept(ck::resolve_variant(
                    data_type_t::bf16, data_type_t::bf16,
                    data_type_t::bf16)),
                "resolve_variant must be noexcept");
}

// ──────────────────────────────────────────────────────────────────
// C.5 — Exhaustive 5-arg negative sweep.  The 5-arg
// `resolve_variant` MUST accept exactly four positive rows:
//   * (bf16, s8, bf16, dynamic_quant=true, compute=s8) → kS8_S8_BF16_SYM
//   * (bf16, s8, bf16, dynamic_quant=true, compute=u8) → kU8_S8_BF16_ASYM
//   * (bf16, s8, f32 , dynamic_quant=true, compute=s8) → kS8_S8_F32_SYM
//   * (bf16, s8, f32 , dynamic_quant=true, compute=u8) → kU8_S8_F32_ASYM
// Everything else in the (data_type_t)^3 × {true,false} ×
// (data_type_t) space must resolve to kUnsupported.  Sweeping
// the full Cartesian product locks the truth table so any future
// new dtype enum value or new variant accidentally weakens the
// gate gets caught by this test instead of leaking into a runtime
// silent misroute.
// ──────────────────────────────────────────────────────────────────
TEST(CkResolveVariantInt8, ExhaustiveNegativeSweep) {
  int n_int8_accepted = 0;
  int n_rejected = 0;
  for (auto src : kAllDtypes) {
    for (auto wei : kAllDtypes) {
      for (auto dst : kAllDtypes) {
        for (bool dq : {false, true}) {
          for (auto cmp : kAllDtypes) {
            const auto v = ck::resolve_variant(src, wei, dst, dq, cmp);
            // The DQ-INT8 family reaches the CK int8 microkernel via TWO
            // src forms (mirror of `resolve_variant`'s grouped-s8
            // acceptance):
            //   * runtime hoist     — src=bf16 with dynamic_quant=true;
            //   * grouped pre-quant — src=s8 (any dynamic_quant, since
            //     group_dynamic_quant CLEARS the flag).
            // Both require wei=s8, dst in {bf16,f32}, compute in {s8,u8}.
            const bool dq_int8_src =
                (src == data_type_t::bf16 && dq)
                || (src == data_type_t::s8);
            const bool int8_family =
                dq_int8_src
                && (wei == data_type_t::s8)
                && (dst == data_type_t::bf16 || dst == data_type_t::f32)
                && (cmp == data_type_t::s8 || cmp == data_type_t::u8);
            if (int8_family) {
              const auto expected = (dst == data_type_t::bf16)
                  ? (cmp == data_type_t::s8
                         ? ck::KernelVariant::kS8_S8_BF16_SYM
                         : ck::KernelVariant::kU8_S8_BF16_ASYM)
                  : (cmp == data_type_t::s8
                         ? ck::KernelVariant::kS8_S8_F32_SYM
                         : ck::KernelVariant::kU8_S8_F32_ASYM);
              EXPECT_EQ(v, expected)
                  << "int8 family must accept src=" << ck_test::dt_name(src)
                  << " wei=" << ck_test::dt_name(wei)
                  << " dst=" << ck_test::dt_name(dst)
                  << " dq=" << dq << " cmp=" << ck_test::dt_name(cmp);
              ++n_int8_accepted;
            } else if (!dq) {
              // Non-int8-family with dynamic_quant=false must mirror the
              // 3-arg overload exactly (compute is ignored off the int8
              // path → bf16/bf16 family or kUnsupported).
              const auto v3 = ck::resolve_variant(src, wei, dst);
              EXPECT_EQ(v, v3)
                  << "dynamic_quant=false non-int8 must mirror the 3-arg "
                     "overload; src=" << ck_test::dt_name(src)
                  << " wei=" << ck_test::dt_name(wei)
                  << " dst=" << ck_test::dt_name(dst)
                  << " cmp=" << ck_test::dt_name(cmp);
              if (v == ck::KernelVariant::kUnsupported) ++n_rejected;
            } else {
              // dynamic_quant=true, not int8 family → must reject.
              EXPECT_EQ(v, ck::KernelVariant::kUnsupported)
                  << "dynamic_quant=true must reject "
                  << ck_test::dt_name(src) << ","
                  << ck_test::dt_name(wei) << ","
                  << ck_test::dt_name(dst) << ",cmp="
                  << ck_test::dt_name(cmp);
              ++n_rejected;
            }
          }
        }
      }
    }
  }
  // Served int8 set = {dst in (bf16,f32)} x {cmp in (s8,u8)} = 4 shapes,
  // reached by 3 (src,dq) forms — (bf16,dq=true), (s8,dq=true),
  // (s8,dq=false) — so 4 x 3 = 12 accepts (4 runtime-hoist + 8 grouped).
  EXPECT_EQ(n_int8_accepted, 12)
      << "Truth table should accept the 4 int8 shapes via the 3 "
         "(src,dynamic_quant) forms (runtime-hoist + grouped pre-quant)";
  EXPECT_GT(n_rejected, 0);
}

}  // namespace
