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

/// @file reorder_test_common.hpp
/// @brief Shared fixture for the reorder gtest surface.
///
/// The reorder tests were split out of the monolithic `test_reorder.cpp`
/// into a self-contained `reorder/` folder (mirroring `group_matmul/`).
/// All reorder `TEST_P` bodies share a single `TestReorder` fixture and the
/// `Reorder` instantiation prefix, so the fixture definition lives here and
/// is included by every `reorder/test_*.cpp`.  The single
/// `INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder, ...)` lives in
/// `test_reorder_regular.cpp`.
///
/// File layout (one suite, split by behavior):
///   * test_reorder_regular.cpp       legacy regular reorder + matmul tests
///                                    (always SKIPPED on the LOWOHA-only path)
///                                    + the lone INSTANTIATE_TEST_SUITE_P
///   * test_static_quant_dequant.cpp  LOWOHA static quant/dequant round-trips
///   * test_dynamic_quant.cpp         LOWOHA dynamic-quant round-trips
///   * test_strided_cases.cpp         LOWOHA strided quant/conversion
///   * test_conversion.cpp            LOWOHA type-conversion round-trips

#ifndef ZENDNNL_GTESTS_REORDER_REORDER_TEST_COMMON_HPP
#define ZENDNNL_GTESTS_REORDER_REORDER_TEST_COMMON_HPP

#include <gtest/gtest.h>
#include <cmath>
#include "gtest_utils.hpp"
// `ReorderType` + the reorder kernel/compare/shape helpers used by every
// reorder TEST_P body live here (lifted out of gtest_utils.hpp).
#include "reorder_test_helpers.hpp"

/** @brief TestReorder is a test class to handle parameters */
class TestReorder : public ::testing::TestWithParam<ReorderType> {
 protected:
  /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixtures i.e. handling the common part of
   *  each test.
   *
   * */
  virtual void SetUp() {
    ReorderType params = GetParam();
    use_LOWOHA = params.is_lowoha_test;

    // LOWOHA-only mode: tests are masked when the user explicitly selects the
    // regular (non-LOWOHA) API. Skip with a message asking the user to use the
    // LOA (LOWOHA) API. GTEST_SKIP() returns from SetUp(), so the LOWOHA
    // initialization below only executes on the LOWOHA path.
    if (!use_LOWOHA) {
      GTEST_SKIP() << "Skipping: please use LOA (LOWOHA) API. "
                   << "Omit --lowoha or pass --lowoha true to run these tests.";
    }
    // Reseed per test so the in-body data fills are reproducible and
    // order-independent for a fixed --seed. Sub-mode selections live on the
    // ReorderType param, so this does not reduce their diversity. After the
    // skip guard so a skipped test doesn't perturb the RNG for later tests.
    srand(static_cast<unsigned int>(seed));
    lowoha_params = params;
    omp_set_num_threads(lowoha_params.num_threads);
  }

  /** @brief TearDown is used to free resource used in test */
  virtual void TearDown() {
    clear_matmul_test_caches();
  }

  uint64_t m, k, n;
  bool transA, transB;
  std::vector<post_op_type_t> po_types;
  bool inplace_reorder;
  data_type_t source_dtype;
  bool use_LOWOHA;
  matmul_algo_t algo;
  int32_t num_threads;
  tensor_factory_t tensor_factory{};

  ReorderType lowoha_params;
};

#endif  // ZENDNNL_GTESTS_REORDER_REORDER_TEST_COMMON_HPP
