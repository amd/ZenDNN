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
  // U8 has no instantiated variant.
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

}  // namespace
