/********************************************************************************
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

#ifndef GTEST_UTILS_GEMV_AI_HPP
#define GTEST_UTILS_GEMV_AI_HPP

#include "../gtest_utils_ai.hpp"

namespace ai_gtests {

class GemvMaxTestCases {
 public:
  inline static int RANDOM_STRESS = 30;
};

void initialize_gemv_nightly_config();

class GemvParameterGenerator {
 public:
  static std::vector<MatmulParamsAI> generate_comprehensive_test_suite();
  static std::vector<MatmulParamsAI> generate_minimal_test_suite();
  static std::vector<MatmulParamsAI> generate_category_specific_params(
    TestCategory category);

 private:
  static void add_kc_path_params(std::vector<MatmulParamsAI> &params);
  static void add_looper_path_params(std::vector<MatmulParamsAI> &params);
  static void add_random_stress_params(std::vector<MatmulParamsAI> &params);
  static void add_boundary_params(std::vector<MatmulParamsAI> &params);
  static void add_edge_case_params(std::vector<MatmulParamsAI> &params);

  static MatmulParamsAI create_gemv_param(
    uint64_t n, uint64_t k,
    DataTypeCombination data_types,
    TestCategory category,
    const PostOpConfig &post_op_config,
    bool trans_b = false,
    bool expect_success = true,
    const std::string &suite_name = "");
};

} // namespace ai_gtests

#endif // GTEST_UTILS_GEMV_AI_HPP
