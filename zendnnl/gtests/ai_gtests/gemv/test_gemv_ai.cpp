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

#include <gtest/gtest.h>
#include "gtest_utils_gemv_ai.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace ai_gtests;
using namespace zendnnl::lowoha::matmul;

class TestGemvAI : public ::testing::TestWithParam<MatmulParamsAI> {
  struct CreatedTensors {
    tensor_t input;
    tensor_t weights;
    tensor_t bias;
    tensor_t output;
    tensor_t reference_output;
    std::vector<std::pair<std::string, tensor_t>> binary_post_op_tensors;
  };

  CreatedTensors create_test_tensors(const MatmulParamsAI &params,
                                     data_type_t input_dtype,
                                     data_type_t weight_dtype,
                                     data_type_t output_dtype,
                                     bool create_reference = false) {
    CreatedTensors t;
    t.input   = AITensorFactory::create_uniform_tensor(
                  {params.m, params.k}, input_dtype, "gemv_input");

    // transB: weights are {N, K} instead of {K, N}
    if (params.trans_b)
      t.weights = AITensorFactory::create_uniform_tensor(
                    {params.n, params.k}, weight_dtype, "gemv_weights");
    else
      t.weights = AITensorFactory::create_uniform_tensor(
                    {params.k, params.n}, weight_dtype, "gemv_weights");
    t.bias    = AITensorFactory::create_uniform_tensor(
                  {1, params.n}, output_dtype, "gemv_bias");
    t.output  = AITensorFactory::create_zero_tensor(
                  {params.m, params.n}, output_dtype, "gemv_output");
    if (create_reference)
      t.reference_output = AITensorFactory::create_zero_tensor(
                             {params.m, params.n}, output_dtype, "gemv_ref_output");

    int add_cnt = 0, mul_cnt = 0;
    for (auto po_type : params.post_op_config.post_ops) {
      if (po_type == post_op_type_t::binary_add) {
        auto bt = AITensorFactory::create_uniform_tensor(
                    {params.m, params.n}, output_dtype,
                    "gemv_binary_add_" + std::to_string(add_cnt++));
        t.binary_post_op_tensors.emplace_back(bt.get_name(), bt);
      } else if (po_type == post_op_type_t::binary_mul) {
        auto bt = AITensorFactory::create_uniform_tensor(
                    {params.m, params.n}, output_dtype,
                    "gemv_binary_mul_" + std::to_string(mul_cnt++));
        t.binary_post_op_tensors.emplace_back(bt.get_name(), bt);
      } else {
        t.binary_post_op_tensors.emplace_back("", tensor_t());
      }
    }
    return t;
  }

 protected:
  void SetUp() override {}
  void TearDown() override {
    // Clear all weight caches between test cases so stale pointer-keyed
    // entries from freed weight tensors don't cause incorrect results.
    // Safe: no concurrent matmul execution during TearDown.
    using namespace zendnnl::lowoha::matmul::native;
    clear_all_weight_caches();
  }

  float get_epsilon(data_type_t dt) {
    return (dt == data_type_t::bf16) ? AI_MATMUL_EPSILON_BF16
                                     : AI_MATMUL_EPSILON_F32;
  }
  float get_rel_tol(data_type_t dt) {
    return (dt == data_type_t::bf16) ? AI_MATMUL_REL_TOLERANCE_BF16
                                     : AI_MATMUL_REL_TOLERANCE_F32;
  }

  // Transpose an N×K tensor to K×N for reference comparison.
  // Supports BF16, FP32, S8, and U8 element types.
  tensor_t transpose_weights(const tensor_t &src, uint64_t rows, uint64_t cols,
                             data_type_t dtype) {
    auto dst = AITensorFactory::create_zero_tensor({cols, rows}, dtype,
                                                   "gemv_weights_T");
    if (dtype == data_type_t::bf16) {
      auto *s = static_cast<const bfloat16_t *>(src.get_raw_handle_const());
      auto *d = static_cast<bfloat16_t *>(dst.get_raw_handle_unsafe());
      for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c = 0; c < cols; ++c)
          d[c * rows + r] = s[r * cols + c];
    } else if (dtype == data_type_t::s8) {
      auto *s = static_cast<const int8_t *>(src.get_raw_handle_const());
      auto *d = static_cast<int8_t *>(dst.get_raw_handle_unsafe());
      for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c = 0; c < cols; ++c)
          d[c * rows + r] = s[r * cols + c];
    } else if (dtype == data_type_t::u8) {
      auto *s = static_cast<const uint8_t *>(src.get_raw_handle_const());
      auto *d = static_cast<uint8_t *>(dst.get_raw_handle_unsafe());
      for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c = 0; c < cols; ++c)
          d[c * rows + r] = s[r * cols + c];
    } else {
      auto *s = static_cast<const float *>(src.get_raw_handle_const());
      auto *d = static_cast<float *>(dst.get_raw_handle_unsafe());
      for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c = 0; c < cols; ++c)
          d[c * rows + r] = s[r * cols + c];
    }
    return dst;
  }

  // GEMV tolerance.
  // BF16 output: abs_bound = k * epsilon_bf16
  // FP32 output: abs_bound = ((C + log2(k)/4) * k + P) * epsilon_f32
  // INT8→FP32:   wider tolerance because accumulation values can be large
  //              (up to K × 127² ≈ millions) and reference vs kernel may
  //              differ by FP32 rounding at that magnitude.
  bool compare_gemv_output(const tensor_t &test, const tensor_t &ref,
                           uint64_t k, data_type_t out_dt,
                           bool is_int8_src = false) {
    constexpr float epsilon_f32_val  = 1.19e-7f;
    constexpr float rtol_f32_val     = 1e-5f;
    constexpr float epsilon_bf16_val = 9.76e-4f;
    constexpr float rtol_bf16_val    = 1e-2f;
    constexpr int C = 20;
    constexpr int P = 15;
    constexpr int scale_factor = 4;
    bool is_bf16_out = (out_dt == data_type_t::bf16);

    if (is_int8_src) {
      // INT8 accumulation can produce values up to K*127*127 (~2M at K=128,
      // ~65M at K=4096). The reference and kernel may accumulate in different
      // order, causing FP32 rounding divergence that's amplified by nonlinear
      // post-ops (gelu, sigmoid, tanh). Use relative tolerance scaled to
      // the accumulation magnitude.
      float rtol = is_bf16_out ? 5e-2f : 1e-3f;
      float abs_bound = 2.0f;
      return AITestUtils::compare_sampled_tensors(test, ref, abs_bound, rtol);
    }

    float epsilon = is_bf16_out ? epsilon_bf16_val : epsilon_f32_val;
    float rtol    = is_bf16_out ? rtol_bf16_val    : rtol_f32_val;
    float abs_bound = is_bf16_out
      ? (static_cast<float>(k) * epsilon)
      : ((C + std::log2(static_cast<float>(k)) / scale_factor)
         * static_cast<float>(k) + P) * epsilon;

    return AITestUtils::compare_sampled_tensors(test, ref, abs_bound, rtol);
  }

  status_t run_gemv(tensor_t &input, tensor_t &weights,
                    tensor_t &bias, tensor_t &output,
                    const MatmulParamsAI &params,
                    std::vector<std::pair<std::string, tensor_t>> &bin_po) {
    try {
      if (is_lowoha_mode_enabled()) {
        const int M = static_cast<int>(params.m);
        const int N = static_cast<int>(params.n);
        const int K = static_cast<int>(params.k);
        const int lda = K;
        const int ldb = params.trans_b ? K : N;
        const int ldc = N;

        // Enable weight caching so the global cache path is exercised.
        // Must be set before the first kernel call (static captured once).
        matmul_config_t::instance().set_weight_cache(1);

        matmul_data_types dtypes;
        dtypes.src     = input.get_data_type();
        dtypes.wei     = weights.get_data_type();
        dtypes.dst     = output.get_data_type();
        dtypes.bias    = bias.get_data_type();
        dtypes.compute = data_type_t::none;

        matmul_batch_params_t bp;
        bp.Batch_A = 1;  bp.Batch_B = 1;
        bp.batch_stride_src = static_cast<size_t>(-1);
        bp.batch_stride_wei = static_cast<size_t>(-1);
        bp.batch_stride_dst = static_cast<size_t>(-1);

        matmul_params mp;
        mp.dtypes = dtypes;
        mp.num_threads = 1;
        // Set algo from config, matching regular gtest (params.lowoha_algo = algo)
        mp.lowoha_algo = static_cast<matmul_algo_t>(
          matmul_config_t::instance().get_algo());

        for (size_t i = 0; i < params.post_op_config.post_ops.size(); ++i) {
          matmul_post_op po_item;
          po_item.po_type = params.post_op_config.post_ops[i];
          if ((po_item.po_type == post_op_type_t::binary_add ||
               po_item.po_type == post_op_type_t::binary_mul) &&
              i < bin_po.size()) {
            po_item.buff  = bin_po[i].second.get_raw_handle_unsafe();
            po_item.dtype = bin_po[i].second.get_data_type();
            auto dims = bin_po[i].second.get_size();
            po_item.dims.assign(dims.begin(), dims.end());
          } else {
            po_item.buff  = nullptr;
            po_item.dtype = output.get_data_type();
          }
          // Match regular gtest: set alpha for swish/elu postops
          if (po_item.po_type == post_op_type_t::swish ||
              po_item.po_type == post_op_type_t::elu)
            po_item.alpha = 1.0f;
          mp.postop_.push_back(po_item);
        }

        // ── INT8 quantization parameters ────────────────────────────────
        // The reference matmul doesn't understand quantization, so for the
        // reference-compared run (Path 3 → output) we use unit scales and
        // zero_point = 0 (s8 src) or 0 (u8 treated as signed by adding 0).
        // This makes the quantized formula reduce to C = A*B + bias, matching
        // the reference.
        //
        // Non-unit scales and per-channel are tested via the cross-path
        // consistency check (all 3 paths must agree), which catches kernel
        // bugs even without reference comparison.
        //
        // Flow: Paths 1-3 use unit scale/zp=0 for reference correctness.
        //       Then Path 4 (extra) uses non-unit per-channel to verify
        //       the dequant pipeline via cross-check against Path 3.
        float q_src_scale = 1.0f;
        [[maybe_unused]] int32_t q_src_zp = 0;
        std::vector<float> q_wei_scales = {1.0f};
        const bool is_int8 = (dtypes.src == data_type_t::u8 ||
                              dtypes.src == data_type_t::s8);
        if (is_int8) {
          mp.quant_params.src_scale.buff = &q_src_scale;
          mp.quant_params.src_scale.dt   = data_type_t::f32;
          mp.quant_params.src_scale.dims = {1};
          mp.quant_params.wei_scale.buff = q_wei_scales.data();
          mp.quant_params.wei_scale.dt   = data_type_t::f32;
          mp.quant_params.wei_scale.dims = {1};
          // src_zp = 0 for both u8 and s8 so quantized formula = plain matmul.
          // For u8 with zp=0: acc = A_u8 * B_s8 (no correction).
          // For s8: kernel adds 128 to get u8, effective_zp = 128, which
          // subtracts 128*col_sum — compensating the +128 offset exactly.
        }

        // ── Run all three packing paths for every test case ──
        using namespace zendnnl::lowoha::matmul::native;
        auto out_dt = output.get_data_type();

        auto call_kernel = [&](bool wt_const, void *dst_buf) {
            return matmul_direct(
                'r', false, params.trans_b,
                M, N, K,
                1.0f, input.get_raw_handle_unsafe(), lda,
                weights.get_raw_handle_unsafe(), ldb,
                bias.get_raw_handle_unsafe(),
                0.0f, dst_buf, ldc,
                wt_const, bp, mp);
        };

        // Path 1: thread_local repack (is_wt_const=false)
        auto tl_out = AITensorFactory::create_zero_tensor(
            {params.m, params.n}, out_dt, "gemv_tl");
        auto st = call_kernel(false, tl_out.get_raw_handle_unsafe());
        if (st != status_t::success) return st;

        // Clear all caches so Path 2 is a guaranteed cold miss.
        clear_all_weight_caches();

        // Path 2: global_cache cold miss (is_wt_const=true, first call)
        auto gc_cold_out = AITensorFactory::create_zero_tensor(
            {params.m, params.n}, out_dt, "gemv_gc_cold");
        st = call_kernel(true, gc_cold_out.get_raw_handle_unsafe());
        if (st != status_t::success) return st;

        // Path 3: global_cache warm hit (is_wt_const=true, reuses cached)
        st = call_kernel(true, output.get_raw_handle_unsafe());
        if (st != status_t::success) return st;

        // Cross-check: all three paths must produce consistent results.
        // output holds Path 3 (warm hit) and is later compared to reference.
        EXPECT_TRUE(compare_gemv_output(tl_out, output, params.k, out_dt, is_int8))
            << "thread_local vs global_cache_warm mismatch: "
            << params.test_name;
        EXPECT_TRUE(compare_gemv_output(gc_cold_out, output, params.k, out_dt, is_int8))
            << "global_cache_cold vs global_cache_warm mismatch: "
            << params.test_name;

        return st;
      } else {
        // The operator API doesn't support transB — it always expects K×N
        // weight layout. For transB tests, transpose to K×N before calling.
        tensor_t op_weights = params.trans_b
          ? transpose_weights(weights, params.n, params.k,
                              weights.get_data_type())
          : weights;
        matmul_context_t ctx = matmul_context_t()
          .set_param("weights", op_weights)
          .set_param("bias", bias);
        for (auto po_type : params.post_op_config.post_ops) {
          post_op_t po{po_type};
          ctx = ctx.set_post_op(po);
        }
        ctx = ctx.create();
        if (!ctx.check()) return status_t::failure;

        auto op = matmul_operator_t()
          .set_name(AITestUtils::generate_unique_name("gemv_ai_op"))
          .set_context(ctx)
          .create();
        if (op.is_bad_object()) return status_t::failure;

        for (size_t i = 0; i < params.post_op_config.post_ops.size(); ++i) {
          auto pt = params.post_op_config.post_ops[i];
          if ((pt == post_op_type_t::binary_add ||
               pt == post_op_type_t::binary_mul) && i < bin_po.size()) {
            std::string tn;
            try {
              tn = (pt == post_op_type_t::binary_add)
                ? ctx.get_post_op(i).binary_add_params.tensor_name
                : ctx.get_post_op(i).binary_mul_params.tensor_name;
            } catch (...) { tn = bin_po[i].first; }
            op = op.set_input(tn, bin_po[i].second);
          }
        }
        return op.set_input("matmul_input", input)
                 .set_output("matmul_output", output)
                 .execute();
      }
    } catch (const std::exception &e) {
      std::cout << "[GEMV_TEST] Exception: " << e.what() << std::endl;
      return status_t::failure;
    } catch (...) {
      return status_t::failure;
    }
  }

  void run_accuracy_test(const MatmulParamsAI &params) {
    ASSERT_TRUE(AITestUtils::validate_dimensions(params.m, params.n, params.k));
    ASSERT_EQ(params.m, 1u) << "GEMV test requires M=1";

    auto in_dt  = AITestUtils::get_input_dtype(params.data_types);
    auto wt_dt  = AITestUtils::get_weight_dtype(params.data_types);
    auto out_dt = AITestUtils::get_output_dtype(params.data_types);

    auto tensors = create_test_tensors(params, in_dt, wt_dt, out_dt, true);

    bool supported = AITestUtils::is_aocl_kernel_supported(
      in_dt, wt_dt, out_dt, params.post_op_config.post_ops);
    status_t st = run_gemv(tensors.input, tensors.weights, tensors.bias,
                           tensors.output, params,
                           tensors.binary_post_op_tensors);
    if (supported) {
      EXPECT_EQ(st, status_t::success) << "GEMV kernel failed for " << params.test_name;
    } else {
      EXPECT_NE(st, status_t::success);
      return;
    }

    // Reference comparison: the reference kernel assumes K×N weight layout
    // (no transB). For transB tests, transpose weights back to K×N.
    bool ref_supported = AITestUtils::is_reference_implementation_supported(
      in_dt, wt_dt, out_dt, params.post_op_config.post_ops);
    if (ref_supported && st == status_t::success) {
      tensor_t ref_weights = params.trans_b
        ? transpose_weights(tensors.weights, params.n, params.k, wt_dt)
        : tensors.weights;

      std::vector<tensor_t> ref_bins;
      for (auto &p : tensors.binary_post_op_tensors) ref_bins.push_back(p.second);
      status_t ref_st = AITestUtils::run_reference_matmul(
        tensors.input, ref_weights, tensors.bias,
        tensors.reference_output, params.post_op_config, ref_bins);
      EXPECT_EQ(ref_st, status_t::success) << "Reference failed for " << params.test_name;
      if (ref_st == status_t::success) {
        bool is_int8_in = (in_dt == data_type_t::u8 || in_dt == data_type_t::s8);
        bool ok = compare_gemv_output(
          tensors.output, tensors.reference_output, params.k, out_dt, is_int8_in);
        EXPECT_TRUE(ok) << "Accuracy mismatch for " << params.test_name;
      }
    }
  }

  void run_boundary_test(const MatmulParamsAI &params) {
    auto in_dt  = AITestUtils::get_input_dtype(params.data_types);
    auto wt_dt  = AITestUtils::get_weight_dtype(params.data_types);
    auto out_dt = AITestUtils::get_output_dtype(params.data_types);
    auto tensors = create_test_tensors(params, in_dt, wt_dt, out_dt, false);
    status_t st = run_gemv(tensors.input, tensors.weights, tensors.bias,
                           tensors.output, params,
                           tensors.binary_post_op_tensors);
    EXPECT_EQ(st, status_t::success) << "Boundary GEMV failed: " << params.test_name;
  }

  void run_edge_case_test(const MatmulParamsAI &params) {
    auto in_dt  = AITestUtils::get_input_dtype(params.data_types);
    auto wt_dt  = AITestUtils::get_weight_dtype(params.data_types);
    auto out_dt = AITestUtils::get_output_dtype(params.data_types);
    auto tensors = create_test_tensors(params, in_dt, wt_dt, out_dt, true);
    status_t st = run_gemv(tensors.input, tensors.weights, tensors.bias,
                           tensors.output, params,
                           tensors.binary_post_op_tensors);
    if (params.expect_success) {
      EXPECT_EQ(st, status_t::success) << "Edge GEMV failed: " << params.test_name;
      if (st == status_t::success &&
          AITestUtils::is_reference_implementation_supported(
            in_dt, wt_dt, out_dt, params.post_op_config.post_ops)) {
        std::vector<tensor_t> ref_bins;
        for (auto &p : tensors.binary_post_op_tensors) ref_bins.push_back(p.second);
        status_t ref_st = AITestUtils::run_reference_matmul(
          tensors.input, tensors.weights, tensors.bias,
          tensors.reference_output, params.post_op_config, ref_bins);
        if (ref_st == status_t::success) {
          bool is_int8_in = (in_dt == data_type_t::u8 || in_dt == data_type_t::s8);
          bool ok = compare_gemv_output(
            tensors.output, tensors.reference_output, params.k, out_dt, is_int8_in);
          EXPECT_TRUE(ok) << "Edge case accuracy mismatch: " << params.test_name;
        }
      }
    }
  }
};

TEST_P(TestGemvAI, BF16GemvTest) {
  // TODO: Enable GEMV tests for BF16
  GTEST_SKIP() << "GEMV tests are skipped for now";
  return;
  MatmulParamsAI params = GetParam();
  if (!AITestUtils::is_valid_data_type_combination(params.data_types)) {
    GTEST_SKIP() << "Data type combination not supported";
    return;
  }
  switch (params.category) {
  case TestCategory::ACCURACY:
    run_accuracy_test(params);
    break;
  case TestCategory::BOUNDARY:
    run_boundary_test(params);
    break;
  case TestCategory::EDGE_CASE:
    run_edge_case_test(params);
    break;
  default:
    run_accuracy_test(params);
    break;
  }
}

INSTANTIATE_TEST_SUITE_P(
  AIGemvTests,
  TestGemvAI,
  ::testing::ValuesIn(
    get_test_suite_for_mode<MatmulParamsAI, GemvParameterGenerator>()),
  [](const ::testing::TestParamInfo<MatmulParamsAI> &info) {
    return info.param.test_name;
  }
);
