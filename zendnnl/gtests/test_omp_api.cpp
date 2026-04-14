/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <atomic>
#include <cstdint>

#include <gtest/gtest.h>

#include "lowoha_operators/common/omp_thread_control.hpp"

namespace zendnnl {
namespace lowoha {
namespace {

// Capture the helper's cached baseline before tests mutate the OMP ICV.
const int32_t kCachedMaxThreads = thread_guard::max_threads();
const int32_t kInitialMaxActiveLevels = omp_get_max_active_levels();

int32_t alternate_thread_count(int32_t current) {
  if (current > 1) {
    return 1;
  }

  return static_cast<int32_t>(omp_get_thread_limit()) > 1 ? 2 : 1;
}

class OmpApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dynamic_before_ = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(kCachedMaxThreads);
    omp_set_max_active_levels(kInitialMaxActiveLevels);
  }

  void TearDown() override {
    omp_set_num_threads(kCachedMaxThreads);
    omp_set_max_active_levels(kInitialMaxActiveLevels);
    omp_set_dynamic(dynamic_before_);
  }

 private:
  int dynamic_before_ = 0;
};

TEST_F(OmpApiTest, ResolveNumThreadsUsesCurrentMaxForAuto) {
  EXPECT_EQ(resolve_num_threads(0, kCachedMaxThreads), kCachedMaxThreads);
}

TEST_F(OmpApiTest, ResolveNumThreadsPreservesExplicitValue) {
  EXPECT_EQ(resolve_num_threads(7, kCachedMaxThreads), 7);
}

TEST_F(OmpApiTest, ResolveNumThreadsSingleThreadPath) {
  EXPECT_EQ(resolve_num_threads(1, kCachedMaxThreads), 1);
}

TEST_F(OmpApiTest, ResolveNumThreadsRequestedMatchesCurrentMax) {
  EXPECT_EQ(resolve_num_threads(kCachedMaxThreads, kCachedMaxThreads),
            kCachedMaxThreads);
}

TEST_F(OmpApiTest, ResolveNumThreadsAutoOnSingleCoreFallback) {
  EXPECT_EQ(resolve_num_threads(0, 1), 1);
}

TEST_F(OmpApiTest, ResolveNumThreadsNegativePassesThrough) {
  EXPECT_EQ(resolve_num_threads(-1, kCachedMaxThreads), -1);
}

TEST_F(OmpApiTest, MaxThreadsIsStableAcrossRepeatedCalls) {
  const int32_t a = thread_guard::max_threads();
  const int32_t b = thread_guard::max_threads();
  const int32_t c = thread_guard::max_threads();
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, c);
}

TEST_F(OmpApiTest, TwoArgumentThreadGuardDoesNotModifyWhenDesiredEqualsCurrent) {
  ASSERT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  {
    thread_guard guard(kCachedMaxThreads, kCachedMaxThreads);
    EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  }
  EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
}

TEST_F(OmpApiTest, MaxThreadsReturnsCachedBaselineAfterIcvChange) {
  const int32_t alternate = alternate_thread_count(kCachedMaxThreads);
  if (alternate == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  omp_set_num_threads(alternate);
  ASSERT_EQ(omp_get_max_threads(), alternate);
  EXPECT_EQ(thread_guard::max_threads(), kCachedMaxThreads);
}

TEST_F(OmpApiTest, TwoArgumentThreadGuardSetsAndRestoresTopLevelIcv) {
  const int32_t alternate = alternate_thread_count(kCachedMaxThreads);
  if (alternate == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  ASSERT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  {
    thread_guard guard(alternate, kCachedMaxThreads);
    EXPECT_EQ(omp_get_max_threads(), alternate);
  }
  EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
}

TEST_F(OmpApiTest, SingleArgumentThreadGuardRestoresCapturedCurrentIcv) {
  const int32_t current = alternate_thread_count(kCachedMaxThreads);
  if (current == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  omp_set_num_threads(current);
  ASSERT_EQ(omp_get_max_threads(), current);

  const int32_t desired = alternate_thread_count(current);
  if (desired == current) {
    GTEST_SKIP() << "Unable to choose a second distinct thread count";
  }

  {
    thread_guard guard(desired);
    EXPECT_EQ(omp_get_max_threads(), desired);
  }
  EXPECT_EQ(omp_get_max_threads(), current);
}

TEST_F(OmpApiTest, SingleArgumentThreadGuardRestoresPerTaskIcvInsideParallelRegion) {
  if (static_cast<int32_t>(omp_get_thread_limit()) < 2) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  std::atomic<int> failures{0};
  std::atomic<int> participants{0};

  #pragma omp parallel num_threads(2)
  {
    participants.fetch_add(1, std::memory_order_relaxed);
    const int32_t before = omp_get_max_threads();

    {
      thread_guard guard(1);
      if (omp_get_max_threads() != 1) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (omp_get_max_threads() != before) {
      failures.fetch_add(1, std::memory_order_relaxed);
    }
  }

  EXPECT_EQ(participants.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
}

TEST_F(OmpApiTest, TwoArgumentThreadGuardUsesPreCapturedPerTaskIcvInLoop) {
  if (static_cast<int32_t>(omp_get_thread_limit()) < 2) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  std::atomic<int> failures{0};
  std::atomic<int> participants{0};

  #pragma omp parallel num_threads(2)
  {
    participants.fetch_add(1, std::memory_order_relaxed);
    const int32_t per_task_icv = omp_get_max_threads();
    const int32_t desired = alternate_thread_count(per_task_icv);

    if (desired == per_task_icv) {
      failures.fetch_add(1, std::memory_order_relaxed);
    } else {
      for (int i = 0; i < 10; ++i) {
        thread_guard guard(desired, per_task_icv);
        if (omp_get_max_threads() != desired) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    if (omp_get_max_threads() != per_task_icv) {
      failures.fetch_add(1, std::memory_order_relaxed);
    }
  }

  EXPECT_EQ(participants.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
}

TEST_F(OmpApiTest, NestedThreadGuardsRestoreCorrectly) {
  const int32_t alternate = alternate_thread_count(kCachedMaxThreads);
  if (alternate == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  ASSERT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  {
    thread_guard outer(alternate, kCachedMaxThreads);
    EXPECT_EQ(omp_get_max_threads(), alternate);
    {
      thread_guard inner(kCachedMaxThreads, alternate);
      EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
    }
    EXPECT_EQ(omp_get_max_threads(), alternate);
  }
  EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
}

TEST_F(OmpApiTest, SequentialGuardsRestoreIndependently) {
  const int32_t alternate = alternate_thread_count(kCachedMaxThreads);
  if (alternate == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(omp_get_max_threads(), kCachedMaxThreads);
    {
      thread_guard guard(alternate, kCachedMaxThreads);
      EXPECT_EQ(omp_get_max_threads(), alternate);
    }
    EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  }
}

TEST_F(OmpApiTest, ProductionPatternResolveAndGuard) {
  const int32_t omp_mt = thread_guard::max_threads();

  const int32_t from_auto = resolve_num_threads(0, omp_mt);
  EXPECT_EQ(from_auto, omp_mt);
  {
    thread_guard guard(from_auto, omp_mt);
    EXPECT_EQ(omp_get_max_threads(), omp_mt);
  }
  EXPECT_EQ(omp_get_max_threads(), omp_mt);

  const int32_t explicit_nt = 1;
  const int32_t from_explicit = resolve_num_threads(explicit_nt, omp_mt);
  EXPECT_EQ(from_explicit, 1);
  {
    thread_guard guard(from_explicit, omp_mt);
    EXPECT_EQ(omp_get_max_threads(), 1);
  }
  EXPECT_EQ(omp_get_max_threads(), omp_mt);
}

TEST_F(OmpApiTest, ThreadGuardRestoresAfterDesiredExceedsBaseline) {
  const int32_t over_request = kCachedMaxThreads * 2;
  if (over_request <= kCachedMaxThreads) {
    GTEST_SKIP() << "Cannot construct a value larger than baseline";
  }

  {
    thread_guard guard(over_request, kCachedMaxThreads);
    EXPECT_EQ(omp_get_max_threads(), over_request);
  }
  EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
}

TEST_F(OmpApiTest, CombinedThreadGuardAndScopedActiveLevels) {
  const int32_t alternate = alternate_thread_count(kCachedMaxThreads);
  if (alternate == kCachedMaxThreads) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  const int32_t orig_levels = omp_get_max_active_levels();
  const int32_t alt_levels = (orig_levels == 1) ? 2 : 1;

  {
    const int32_t omp_mt = thread_guard::max_threads();
    const int32_t nt = resolve_num_threads(alternate, omp_mt);
    thread_guard tg(nt, omp_mt);
    scoped_active_levels sal(alt_levels);

    EXPECT_EQ(omp_get_max_threads(), alternate);
    EXPECT_EQ(omp_get_max_active_levels(), alt_levels);
  }

  EXPECT_EQ(omp_get_max_threads(), kCachedMaxThreads);
  EXPECT_EQ(omp_get_max_active_levels(), orig_levels);
}

TEST_F(OmpApiTest, MaxThreadsReturnsSameValueFromAllParallelThreads) {
  if (static_cast<int32_t>(omp_get_thread_limit()) < 2) {
    GTEST_SKIP() << "OpenMP runtime exposes only one usable thread";
  }

  std::atomic<int> mismatches{0};
  std::atomic<int> participants{0};
  const int32_t expected = thread_guard::max_threads();

  #pragma omp parallel num_threads(2)
  {
    participants.fetch_add(1, std::memory_order_relaxed);
    if (thread_guard::max_threads() != expected) {
      mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  }

  EXPECT_EQ(participants.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0);
}

TEST_F(OmpApiTest, ScopedActiveLevelsDoesNotModifyWhenDesiredEqualsCurrent) {
  const int32_t current = omp_get_max_active_levels();
  {
    scoped_active_levels guard(current);
    EXPECT_EQ(omp_get_max_active_levels(), current);
  }
  EXPECT_EQ(omp_get_max_active_levels(), current);
}

TEST_F(OmpApiTest, ScopedActiveLevelsSetsAndRestoresPreviousValue) {
  const int32_t original = omp_get_max_active_levels();
  const int32_t desired = (original == 1) ? 2 : 1;

  {
    scoped_active_levels guard(desired);
    if (omp_get_max_active_levels() != desired) {
      GTEST_SKIP() << "OpenMP runtime did not accept requested active level";
    }
    EXPECT_EQ(omp_get_max_active_levels(), desired);
  }

  EXPECT_EQ(omp_get_max_active_levels(), original);
}

} // namespace
} // namespace lowoha
} // namespace zendnnl
