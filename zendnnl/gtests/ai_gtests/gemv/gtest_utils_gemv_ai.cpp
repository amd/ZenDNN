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

#include "gtest_utils_gemv_ai.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <cstdlib>

namespace ai_gtests {

template std::vector<MatmulParamsAI>
get_test_suite_for_mode<MatmulParamsAI, GemvParameterGenerator>();

void initialize_gemv_nightly_config() {
  GemvMaxTestCases::RANDOM_STRESS = 150;

  // Override via env: GEMV_STRESS_COUNT=N sets random stress tests per combo
  const char *env = std::getenv("GEMV_STRESS_COUNT");
  if (env) {
    try {
      int val = std::stoi(env);
      if (val > 0) GemvMaxTestCases::RANDOM_STRESS = val;
    } catch (...) {}
  }
  std::cout << "[AI_GEMV] Nightly mode: RANDOM_STRESS="
            << GemvMaxTestCases::RANDOM_STRESS << std::endl;
}

namespace {
struct GemvNightlyInitializer {
  GemvNightlyInitializer() {
    if (ai_gtest_mode == TestMode::NIGHTLY)
      initialize_gemv_nightly_config();
  }
};
static GemvNightlyInitializer s_gemv_nightly_init;
} // anonymous namespace

static std::mt19937 gemv_rng(
  std::chrono::steady_clock::now().time_since_epoch().count());

static uint64_t gemv_random_dim(uint64_t min_d, uint64_t max_d) {
  std::uniform_int_distribution<uint64_t> dist(min_d, max_d);
  return dist(gemv_rng);
}

// BF16 GEMV data type combinations (input/weight always bf16 to trigger BRGEMM)
static const std::vector<DataTypeCombination> gemv_bf16_combos = {
  DataTypeCombination::BF16_BF16_BF16,  // BF16 in, BF16 wt → BF16 output
  DataTypeCombination::BF16_BF16_F32,   // BF16 in, BF16 wt → F32 output
};

// INT8 GEMV data type combinations
static const std::vector<DataTypeCombination> gemv_int8_combos = {
  DataTypeCombination::U8_S8_F32,   // u8 src, s8 weight → f32 output
  DataTypeCombination::U8_S8_BF16,  // u8 src, s8 weight → bf16 output
  DataTypeCombination::S8_S8_F32,   // s8 src, s8 weight → f32 output
  DataTypeCombination::S8_S8_BF16,  // s8 src, s8 weight → bf16 output
};

// Activation post-ops that the KC GEMV kernel can fuse
static std::vector<PostOpConfig> get_gemv_postop_configs() {
  std::vector<PostOpConfig> configs;
  configs.push_back(PostOpConfig{});  // none
  configs.push_back(AITestUtils::create_relu_config());
  configs.push_back(AITestUtils::create_gelu_tanh_config());
  configs.push_back(AITestUtils::create_gelu_erf_config());
  configs.push_back(AITestUtils::create_sigmoid_config());
  configs.push_back(AITestUtils::create_tanh_config());
  configs.push_back(AITestUtils::create_silu_config());
  return configs;
}

static std::vector<PostOpConfig> get_gemv_all_postop_configs() {
  return AITestUtils::get_all_post_op_configs();
}

MatmulParamsAI GemvParameterGenerator::create_gemv_param(
  uint64_t n, uint64_t k,
  DataTypeCombination data_types,
  TestCategory category,
  const PostOpConfig &post_op_config,
  bool trans_b,
  bool expect_success,
  const std::string &suite_name) {

  MatmulParamsAI param;
  param.m = 1;
  param.n = n;
  param.k = k;
  param.data_types = data_types;
  param.category = category;
  param.post_op_config = post_op_config;
  param.trans_a = false;
  param.trans_b = trans_b;
  param.expect_success = expect_success;

  static std::atomic<uint64_t> counter{0};
  auto dtype_str = [](data_type_t dt) {
    switch (dt) {
    case data_type_t::f32:   return "f32";
    case data_type_t::bf16:  return "bf16";
    case data_type_t::u8:    return "u8";
    case data_type_t::s8:    return "s8";
    default:                 return "unk";
    }
  };
  std::string in_s  = dtype_str(AITestUtils::get_input_dtype(data_types));
  std::string wt_s  = dtype_str(AITestUtils::get_weight_dtype(data_types));
  std::string out_s = dtype_str(AITestUtils::get_output_dtype(data_types));
  std::string tb_s  = trans_b ? "transB" : "noTransB";

  std::string prefix = suite_name.empty() ? "gemv" : suite_name;
  param.test_name = prefix
    + "_n" + std::to_string(n) + "_k" + std::to_string(k)
    + "_" + in_s + "_" + wt_s + "_" + out_s
    + "_" + tb_s
    + "_" + post_op_config.config_name
    + "_" + std::to_string(counter.fetch_add(1));
  return param;
}

// KC path: N<=256, packed B fits in L2. These shapes hit bf16_gemv_direct.
void GemvParameterGenerator::add_kc_path_params(
    std::vector<MatmulParamsAI> &params) {
  static const uint64_t kc_n_vals[] = {1, 2, 4, 8, 16, 32, 64, 128, 192, 256};
  static const uint64_t kc_k_vals[] = {32, 64, 128, 256, 512, 1024};
  auto postop_cfgs = get_gemv_postop_configs();

  // BF16 KC tests
  for (auto combo : gemv_bf16_combos) {
    for (const auto &po : postop_cfgs) {
      for (auto n : kc_n_vals) {
        for (auto k : kc_k_vals) {
          params.push_back(create_gemv_param(
            n, k, combo, TestCategory::ACCURACY, po,
            false, true, "gemv_kc"));
        }
      }
    }
  }

  // INT8 KC tests (same shapes, all 4 dtype combos)
  for (auto combo : gemv_int8_combos) {
    for (const auto &po : postop_cfgs) {
      for (auto n : kc_n_vals) {
        for (auto k : kc_k_vals) {
          params.push_back(create_gemv_param(
            n, k, combo, TestCategory::ACCURACY, po,
            false, true, "gemv_int8_kc"));
        }
      }
    }
  }

  // transB variants for a subset of shapes
  static const uint64_t tb_n_vals[] = {32, 128, 256};
  static const uint64_t tb_k_vals[] = {64, 256, 512};
  PostOpConfig no_postop;
  for (auto combo : gemv_bf16_combos) {
    for (auto n : tb_n_vals) {
      for (auto k : tb_k_vals) {
        params.push_back(create_gemv_param(
          n, k, combo, TestCategory::ACCURACY, no_postop,
          true, true, "gemv_kc_transB"));
      }
    }
  }
  // INT8 transB
  for (auto combo : gemv_int8_combos) {
    for (auto n : tb_n_vals) {
      for (auto k : tb_k_vals) {
        params.push_back(create_gemv_param(
          n, k, combo, TestCategory::ACCURACY, no_postop,
          true, true, "gemv_int8_kc_transB"));
      }
    }
  }
}

// Looper BRGEMM M=1 path: N>256. These go through bf16_brgemm_execute.
void GemvParameterGenerator::add_looper_path_params(
    std::vector<MatmulParamsAI> &params) {
  static const uint64_t lp_n_vals[] = {257, 384, 512, 768, 1024};
  static const uint64_t lp_k_vals[] = {32, 64, 128, 256, 512, 1024};
  auto postop_cfgs = get_gemv_postop_configs();

  for (auto combo : gemv_bf16_combos) {
    for (const auto &po : postop_cfgs) {
      for (auto n : lp_n_vals) {
        for (auto k : lp_k_vals) {
          params.push_back(create_gemv_param(
            n, k, combo, TestCategory::ACCURACY, po,
            false, true, "gemv_looper"));
        }
      }
    }
  }
  // INT8 looper path: exercises int8_brgemm_execute for large N
  for (auto combo : gemv_int8_combos) {
    for (const auto &po : postop_cfgs) {
      for (auto n : lp_n_vals) {
        for (auto k : lp_k_vals) {
          params.push_back(create_gemv_param(
            n, k, combo, TestCategory::ACCURACY, po,
            false, true, "gemv_int8_looper"));
        }
      }
    }
  }
}

// Random stress: wide N/K range with all postops and transB.
void GemvParameterGenerator::add_random_stress_params(
    std::vector<MatmulParamsAI> &params) {
  auto all_postops = get_gemv_all_postop_configs();
  // GEMV_STRESS_COUNT env var overrides the compiled-in default at any time
  int max_per_combo = GemvMaxTestCases::RANDOM_STRESS;
  const char *env = std::getenv("GEMV_STRESS_COUNT");
  if (env) {
    try { int v = std::stoi(env); if (v > 0) max_per_combo = v; }
    catch (...) {}
  }

  auto add_stress = [&](const std::vector<DataTypeCombination> &combos,
                        const std::string &prefix) {
    for (auto combo : combos) {
      auto in_dt  = AITestUtils::get_input_dtype(combo);
      auto wt_dt  = AITestUtils::get_weight_dtype(combo);
      auto out_dt = AITestUtils::get_output_dtype(combo);
      for (const auto &po : all_postops) {
        if (!AITestUtils::is_aocl_kernel_supported(in_dt, wt_dt, out_dt,
                                                    po.post_ops))
          continue;
        for (int i = 0; i < max_per_combo; ++i) {
          uint64_t n = gemv_random_dim(1, 2048);
          uint64_t k = gemv_random_dim(1, 4096);
          bool tb = (gemv_random_dim(0, 1) == 1);
          params.push_back(create_gemv_param(
            n, k, combo, TestCategory::ACCURACY, po,
            tb, true, prefix));
        }
      }
    }
  };
  add_stress(gemv_bf16_combos, "gemv_stress");

  // INT8 stress: same N/K range as BF16. The KC kernel handles any N
  // where packed B fits in L2 (INT8 is 1 byte/elem → 2x larger N than BF16).
  // Shapes exceeding L2 gracefully fall back to DLP.
  // Use only activation post-ops that the INT8 KC kernel can fuse.
  auto int8_postops = get_gemv_postop_configs();
  for (auto combo : gemv_int8_combos) {
    for (const auto &po : int8_postops) {
      for (int i = 0; i < max_per_combo; ++i) {
        uint64_t n = gemv_random_dim(1, 2048);
        uint64_t k = gemv_random_dim(1, 4096);
        bool tb = (gemv_random_dim(0, 1) == 1);
        params.push_back(create_gemv_param(
          n, k, combo, TestCategory::ACCURACY, po,
          tb, true, "gemv_int8_stress"));
      }
    }
  }
}

void GemvParameterGenerator::add_boundary_params(
    std::vector<MatmulParamsAI> &params) {
  static const std::vector<std::pair<uint64_t, uint64_t>> boundary_nk = {
    {1, 1}, {1, 32}, {32, 1},
    {63, 64}, {64, 63}, {65, 64},    // NR_PACK boundary
    {127, 128}, {128, 127}, {129, 128},
    {255, 256}, {256, 255}, {257, 256},  // KC path N boundary
    {256, 1024}, {1, 4096}, {2048, 1},
  };
  PostOpConfig no_postop;
  for (auto combo : gemv_bf16_combos) {
    for (const auto &[n, k] : boundary_nk) {
      params.push_back(create_gemv_param(
        n, k, combo, TestCategory::BOUNDARY, no_postop,
        false, true, "gemv_boundary"));
    }
  }
  for (auto combo : gemv_int8_combos) {
    for (const auto &[n, k] : boundary_nk) {
      params.push_back(create_gemv_param(
        n, k, combo, TestCategory::BOUNDARY, no_postop,
        false, true, "gemv_int8_boundary"));
    }
  }
}

void GemvParameterGenerator::add_edge_case_params(
    std::vector<MatmulParamsAI> &params) {
  static const std::vector<std::pair<uint64_t, uint64_t>> edge_nk = {
    {1, 1},
    {1, 8192},     // very deep K
    {4096, 1},     // very wide N, tiny K
    {3, 7},        // odd non-power-of-2
    {17, 33},      // prime-ish
    {256, 256},    // KC boundary exact
    {255, 1023},   // just below boundaries
  };
  auto postop_cfgs = get_gemv_postop_configs();
  for (auto combo : gemv_bf16_combos) {
    for (const auto &po : postop_cfgs) {
      for (const auto &[n, k] : edge_nk) {
        params.push_back(create_gemv_param(
          n, k, combo, TestCategory::EDGE_CASE, po,
          false, true, "gemv_edge"));
      }
    }
  }
  for (auto combo : gemv_int8_combos) {
    for (const auto &po : postop_cfgs) {
      for (const auto &[n, k] : edge_nk) {
        params.push_back(create_gemv_param(
          n, k, combo, TestCategory::EDGE_CASE, po,
          false, true, "gemv_int8_edge"));
      }
    }
  }
}

std::vector<MatmulParamsAI>
GemvParameterGenerator::generate_comprehensive_test_suite() {
  std::vector<MatmulParamsAI> all;
  add_kc_path_params(all);
  add_looper_path_params(all);
  add_random_stress_params(all);
  add_boundary_params(all);
  add_edge_case_params(all);
  std::cout << "[AI_GEMV] Generated " << all.size()
            << " GEMV test cases" << std::endl;
  return all;
}

std::vector<MatmulParamsAI>
GemvParameterGenerator::generate_minimal_test_suite() {
  std::vector<MatmulParamsAI> minimal;
  PostOpConfig no_postop;
  PostOpConfig relu_po = AITestUtils::create_relu_config();

  for (auto combo : gemv_bf16_combos) {
    // KC path: a few representative shapes
    minimal.push_back(create_gemv_param(64,  128, combo, TestCategory::ACCURACY, no_postop,  false, true, "gemv_min"));
    minimal.push_back(create_gemv_param(128, 256, combo, TestCategory::ACCURACY, relu_po,    false, true, "gemv_min"));
    minimal.push_back(create_gemv_param(256, 512, combo, TestCategory::ACCURACY, no_postop,  false, true, "gemv_min"));
    // Looper path
    minimal.push_back(create_gemv_param(512, 256, combo, TestCategory::ACCURACY, no_postop,  false, true, "gemv_min"));
    minimal.push_back(create_gemv_param(1024, 64, combo, TestCategory::ACCURACY, relu_po,    false, true, "gemv_min"));
    // transB
    minimal.push_back(create_gemv_param(128, 256, combo, TestCategory::ACCURACY, no_postop,  true,  true, "gemv_min"));
    // Boundary
    minimal.push_back(create_gemv_param(1, 1,     combo, TestCategory::BOUNDARY, no_postop,  false, true, "gemv_min"));
    minimal.push_back(create_gemv_param(256, 256, combo, TestCategory::BOUNDARY, no_postop,  false, true, "gemv_min"));
  }
  std::cout << "[AI_GEMV] Generated " << minimal.size()
            << " minimal GEMV test cases" << std::endl;
  return minimal;
}

std::vector<MatmulParamsAI>
GemvParameterGenerator::generate_category_specific_params(
    TestCategory category) {
  std::vector<MatmulParamsAI> params;
  switch (category) {
  case TestCategory::ACCURACY:
    add_kc_path_params(params);
    add_looper_path_params(params);
    add_random_stress_params(params);
    break;
  case TestCategory::BOUNDARY:
    add_boundary_params(params);
    break;
  case TestCategory::EDGE_CASE:
    add_edge_case_params(params);
    break;
  default:
    break;
  }
  return params;
}

} // namespace ai_gtests
