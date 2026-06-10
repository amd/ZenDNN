/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file group_matmul_test_helpers.cpp
/// @brief Implementations of the group-matmul-specific declarations in
///        `group_matmul_test_helpers.hpp`.  Lifted out of `gtest_utils.cpp`
///        during the gtests folder refactor so the operator-agnostic
///        translation unit stays focused on shared infrastructure.

#include "group_matmul_test_helpers.hpp"

#include <random>
#include <vector>

#include <omp.h>

#include "gtest_utils.hpp"

// ── GroupQuantMatmulType constructor (was gtest_utils.cpp:242-304) ─────────
// GroupQuantMatmulType constructor — constrained-by-construction params for the
// group-matmul quantized test suites.  Picks the same shape/algo/dtype
// dimensions as MatmulType, but pins the destabilizing knobs (transA/transB,
// alpha, beta, post-op chain) to values where activated comparison stays
// well-bounded, and rounds K down to a multiple of 4 so symmetric / dynamic
// INT8 K-grouping doesn't need per-test rounding.  The fixture's
// alpha=1, beta=0, transA=transB=false constants live on
// `TestGroupMatmulQuant`; this struct only carries the random axes.
GroupQuantMatmulType::GroupQuantMatmulType(uint32_t test_index,
    uint32_t total_tests) {
  std::mt19937 gen(rand());
  matmul_m = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  // K aligned to 4 — INT8 SYM/DYNAMIC tests previously rounded internally
  // (`sym_k = (k/4)*4`); centralising the alignment here lets the test
  // bodies use `k` directly.
  uint64_t raw_k = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;
  matmul_k = (raw_k / 4) * 4;
  if (matmul_k == 0) {
    matmul_k = 4;
  }
  matmul_n = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END;

  if (cmd_num_threads) {
    num_threads = cmd_num_threads;
  }
  else {
    int max_threads = omp_get_max_threads();
    std::uniform_int_distribution<int> thread_dist(1, max_threads);
    num_threads = thread_dist(gen);
  }

  // Compatible algo set — exclude LIBXSMM / OneDNN backends when the build
  // didn't include them (mirrors the MatmulType selector logic).  Suite
  // bodies (e.g. WOQ_BF16_S4) still self-skip for paths that the specific
  // dtype/granularity combo doesn't support.
  std::vector<int> algo_list;
  bool onednn_disabled  = !ZENDNNL_DEPENDS_ONEDNN;
  bool libxsmm_disabled = !ZENDNNL_DEPENDS_LIBXSMM;
  if (onednn_disabled && libxsmm_disabled) {
    algo_list = {1, 4};   // aocl_dlp_blocked, aocl_dlp
  }
  else if (onednn_disabled) {
    algo_list = {1, 3, 4, 6};
  }
  else if (libxsmm_disabled) {
    algo_list = {1, 2, 4, 5};
  }
  else {
    algo_list = {1, 2, 3, 4, 5, 6};
  }
  std::uniform_int_distribution<size_t> algo_dist(0, algo_list.size() - 1);
  algo = static_cast<matmul_algo_t>(algo_list[algo_dist(gen)]);

  source_dtype = (rand() % 2 == 0) ? data_type_t::s8 : data_type_t::u8;
  // Output dtype is constrained to f32/bf16 — the gated_act validator in
  // group_matmul_direct rejects quantized dst, and SYM/DYNAMIC tests
  // hardcode their own output dtype anyway, so this drives only the
  // INT8 / WOQ tests' picker.
  output_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
  weight_granularity = (rand() % 2 == 0) ? quant_granularity_t::tensor :
                       quant_granularity_t::channel;
  (void)test_index;
  (void)total_tests;
}

// ── PrintTo(GroupQuantMatmulType) (was gtest_utils.cpp:1384-1392) ──────────
void PrintTo(const GroupQuantMatmulType &value, ::std::ostream *os) {
  *os << "m=" << value.matmul_m << ", k=" << value.matmul_k << ", n="
      << value.matmul_n
      << ", algo=" << algoToStr(value.algo)
      << ", src_dtype=" << dtype_info(value.source_dtype)
      << ", dst_dtype=" << dtype_info(value.output_dtype)
      << ", weight_granularity=" << static_cast<int>(value.weight_granularity)
      << ", num_threads=" << value.num_threads << ", seed=" << seed;
}

// ── group_matmul_kernel_test (was gtest_utils.cpp:2471-2606) ───────────────
status_t group_matmul_kernel_test(
  std::vector<tensor_t> &inputs,
  std::vector<tensor_t> &weights,
  std::vector<tensor_t> &biases,
  std::vector<tensor_t> &outputs,
  matmul_algo_t algo,
  float alpha,
  float beta,
  const group_matmul_moe_postop_params *moe_postop,
  const grp_matmul_gated_act_params *gated_act) {
  using namespace zendnnl::lowoha::matmul;
  try {
    const size_t num_ops = inputs.size();
    if (num_ops == 0 || weights.size() != num_ops ||
        biases.size() != num_ops || outputs.size() != num_ops) {
      log_error("group_matmul_kernel_test: vector size mismatch");
      return status_t::failure;
    }

    std::vector<char>           layouts(num_ops, 'r');
    std::vector<bool>           transAs(num_ops), transBs(num_ops);
    std::vector<int>            Ms(num_ops), Ns(num_ops), Ks(num_ops);
    std::vector<float>          alphas(num_ops, alpha), betas(num_ops, beta);
    std::vector<int>            ldas_v(num_ops), ldbs_v(num_ops), ldcs_v(num_ops);
    std::vector<const void *>   srcs(num_ops), wts(num_ops), bias_ptrs(num_ops);
    std::vector<void *>         dsts(num_ops);
    std::vector<bool>           is_wt_c(num_ops);
    std::vector<matmul_params>  params_v(num_ops);

    for (size_t i = 0; i < num_ops; ++i) {
      if (!inputs[i].check() || !weights[i].check() || !outputs[i].check()) {
        log_error("group_matmul_kernel_test: invalid tensor at expert ", i);
        return status_t::failure;
      }

      transAs[i] = (inputs[i].get_order() == "ba");
      transBs[i] = (weights[i].get_order() == "ba");
      Ms[i] = static_cast<int>(outputs[i].get_size(0));
      Ks[i] = static_cast<int>(inputs[i].get_size(1));
      Ns[i] = static_cast<int>(outputs[i].get_size(1));
      ldas_v[i] = transAs[i] ? inputs[i].get_stride(1)
                  : inputs[i].get_stride(0);
      ldbs_v[i] = transBs[i] ? weights[i].get_stride(1)
                  : weights[i].get_stride(0);
      ldcs_v[i] = outputs[i].get_stride(0);

      data_type_t src_dt  = inputs[i].get_data_type();
      data_type_t wei_dt  = weights[i].get_data_type();
      data_type_t dst_dt  = outputs[i].get_data_type();
      data_type_t bias_dt = biases[i].check() ? biases[i].get_data_type()
                            : data_type_t::f32;

      params_v[i].dtypes.src  = src_dt;
      params_v[i].dtypes.wei  = wei_dt;
      params_v[i].dtypes.dst  = dst_dt;
      params_v[i].dtypes.bias = bias_dt;
      params_v[i].num_threads = 0;
      // Forward the caller-requested algo so the F16-aware setup in
      // group_matmul_direct (ISA gate + reference-accum publication)
      // sees the actual algo the test wants exercised. Prior to this
      // the field was left at matmul_algo_t::none and the production
      // F16 accum-type publication couldn't distinguish AOCL F16 (F16
      // accumulator) from other paths (F32 accumulator), causing
      // kernel vs forced-reference drift for the F16 group-matmul
      // cases. Mirrors how matmul_kernel_test in gtest_utils.cpp
      // populates params.lowoha_algo before invoking matmul_direct.
      params_v[i].lowoha_algo = algo;

      bool is_woq = (src_dt == data_type_t::bf16 &&
                     (wei_dt == data_type_t::s4 || wei_dt == data_type_t::u4));
      bool is_wei_s8 = (wei_dt == data_type_t::s8);
      is_wt_c[i] = is_woq || is_wei_s8;

      auto extract_quant = [](const tensor_t &t,
                              matmul_quantization_params_t::matmul_quant_t &scale_p,
      matmul_quantization_params_t::matmul_quant_t &zp_p) {
        if (!t.is_quantized()) {
          return;
        }
        const void *sb = t.get_quant_scale_raw_handle_const();
        if (sb) {
          scale_p.buff = sb;
          scale_p.dt   = t.get_quant_scale_data_type();
          auto sz = t.get_quant_scale_size();
          scale_p.dims.assign(sz.begin(), sz.end());
        }
        if (t.get_quant_subtype() == quant_subtype_t::asymmetric) {
          const void *zb = t.get_quant_zero_raw_handle_const();
          if (zb) {
            zp_p.buff = zb;
            zp_p.dt   = t.get_quant_zero_data_type();
            auto sz = t.get_quant_zero_size();
            zp_p.dims.assign(sz.begin(), sz.end());
          }
        }
      };

      if (is_woq || is_wei_s8) {
        extract_quant(inputs[i], params_v[i].quant_params.src_scale,
                      params_v[i].quant_params.src_zp);
        extract_quant(weights[i], params_v[i].quant_params.wei_scale,
                      params_v[i].quant_params.wei_zp);
        extract_quant(outputs[i], params_v[i].quant_params.dst_scale,
                      params_v[i].quant_params.dst_zp);
      }

      if (is_wei_s8 &&
          (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32) &&
          inputs[i].is_quantized()) {
        params_v[i].dynamic_quant = true;
        params_v[i].dtypes.compute = data_type_t::s8;
      }

      //TODO: For LIBXSMM matmul, bias is not supported currently due to accuracy issues
      const bool is_libxsmm_kernel = (algo == matmul_algo_t::libxsmm ||
                                      algo == matmul_algo_t::libxsmm_blocked);
      const bool skip_bias =
        (is_libxsmm_kernel && dst_dt == data_type_t::bf16);

      srcs[i]      = inputs[i].get_raw_handle_unsafe();
      wts[i]       = weights[i].get_raw_handle_unsafe();
      bias_ptrs[i] = (skip_bias || !biases[i].check())
                     ? nullptr
                     : static_cast<const void *>(biases[i].get_raw_handle_unsafe());
      dsts[i]      = outputs[i].get_raw_handle_unsafe();
    }

    status_t status = group_matmul_direct(
                        layouts, transAs, transBs, Ms, Ns, Ks, alphas,
                        srcs, ldas_v, wts, ldbs_v, bias_ptrs, betas, dsts, ldcs_v,
                        is_wt_c, params_v, moe_postop, gated_act);

    if (status != status_t::success && status != status_t::isa_unsupported) {
      log_error("group_matmul_kernel_test: group_matmul_direct failed");
    }
    return status;
  }
  catch (const std::exception &e) {
    log_error("group_matmul_kernel_test: ", e.what());
    return status_t::failure;
  }
}
