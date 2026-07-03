/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_matmul_utils.hpp"
#include "ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "partitioning/bmm/looper/bmm_looper.hpp"
#include "partitioning/matmul/matmul_partitioner.hpp"
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_kernel.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"
#include "lowoha_operators/matmul/auto_tuner/auto_tuner.hpp"
#include "matmul_native/native_matmul.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

void matmul_kernel_wrapper(char layout, char transA, char transB,
                           int M, int N, int K,
                           float alpha,
                           const void *A, int lda,
                           const void *B, int ldb,
                           float beta,
                           void *C, int ldc,
                           matmul_data_types &dtypes,
                           zendnnl::ops::matmul_algo_t &kernel,
                           char mem_format_a, char mem_format_b,
                           matmul_params &lowoha_param, matmul_batch_params_t &batch_params,
                           const void *bias, bool is_weights_const,
                           int num_threads) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == matmul_algo_t::libxsmm) {
    log_info("Using libxsmm kernel");
    if (run_libxsmm_std(transA, transB, M, N, K, beta, lda, ldb, ldc, A, B, C,
                        dtypes, lowoha_param, bias)) {
      return;
    }
  }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    log_info("Using onednn kernel");
    matmul_onednn_wrapper(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                          ldc, lowoha_param, batch_params, bias, kernel);
    return;
  }
#endif
  if (kernel == matmul_algo_t::native_gemm ||
      kernel == matmul_algo_t::native_brgemm) {
    // Native kernels support FP32 and BF16 (src=BF16,wei=BF16,dst=BF16|FP32), no transA
    const bool is_fp32 = (lowoha_param.dtypes.src == data_type_t::f32 &&
                          lowoha_param.dtypes.wei == data_type_t::f32 &&
                          lowoha_param.dtypes.dst == data_type_t::f32);
    const bool is_bf16 = (lowoha_param.dtypes.src == data_type_t::bf16 &&
                          lowoha_param.dtypes.wei == data_type_t::bf16 &&
                          (lowoha_param.dtypes.dst == data_type_t::bf16 ||
                           lowoha_param.dtypes.dst == data_type_t::f32));
    if (!is_fp32 && !is_bf16) {
      log_info("Native kernel: unsupported data type, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
    else if (transA == 't') {
      log_info("Native kernel: transA not supported, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
    else {
      apilog_info("Executing matmul LOWOHA kernel with ",
                  kernel_to_string(kernel),
                  ", algo: ", static_cast<int>(kernel));
      bool executed = native::native_matmul_execute(kernel, layout, transA == 't',
                      transB == 't', M, N, K, alpha, A, lda, B, ldb, bias, beta, C, ldc,
                      is_weights_const, num_threads, lowoha_param);
      if (executed) {
        return;
      }
      log_info("Native kernel execution failed, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
  }

#if !ZENDNNL_DEPENDS_AOCLDLP
  // No AOCL-DLP backend in this build. This tail is the universal AOCL
  // fallback: it is reachable not only when an AOCL kernel is selected, but
  // also when another backend falls through without computing (e.g. a libxsmm
  // runtime decline, or onednn selected with oneDNN compiled out). It can also
  // run inside OpenMP parallel regions (group-matmul / BMM), where letting
  // run_dlp throw would call std::terminate. Log and return without computing
  // instead of crashing.
  log_error("Kernel ", kernel_to_string(kernel), " requires the AOCL-DLP "
            "fallback, but ZenDNNL was built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0); matmul output not computed.");
  // kernel is a reference: mark the unavailable AOCL-DLP fallback so
  // matmul_direct()'s post-dispatch guard reports status_t::unimplemented
  // even when the fell-through kernel name was libxsmm/onednn/etc.
  kernel = matmul_algo_t::aocl_dlp;
  return;
#else
  log_info("Using AOCL DLP kernel");
  run_dlp(layout, transA, transB, M, N, K, alpha, beta,
          lda, ldb, ldc, mem_format_a, mem_format_b,
          A, B, C, dtypes, lowoha_param, bias, kernel, is_weights_const);
  return;
#endif
}

void matmul_execute(const char layout,
                    const bool transA, const bool transB,
                    const int M, const int N, const int K, const float alpha,
                    const void *src, const int lda,
                    const void *weight, const int ldb,
                    const void *bias, const float beta,
                    void *dst, const int ldc,
                    const bool is_weights_const,
                    const size_t src_type_size, const size_t out_type_size,
                    const int num_threads,
                    matmul_algo_t &kernel, matmul_params &params,
                    matmul_batch_params_t &batch_params,
                    unsigned int auto_version) {

  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  // Auto-tuner kernel selection for single batch.
  if (kernel == matmul_algo_t::auto_tuner) {
    if (auto_version == 1) {
      kernel = auto_compute_matmul_v1(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const,
                                      num_threads);
    }
    else {
      kernel = auto_compute_matmul_v2(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const,
                                      num_threads);
    }
    return;
  }

// Currently supported only for LIBXSMM BACKEND
  if (should_use_mm_partitioner(kernel)) {
    // Setup partition configuration
    matmul_partition_config_t part_config;
    part_config.M = M;
    part_config.N = N;
    part_config.K = K;
    part_config.num_threads = num_threads;
    part_config.kernel = kernel;
    part_config.src_type_size = src_type_size;
    part_config.out_type_size = out_type_size;
    part_config.lda = lda;
    part_config.ldb = ldb;
    part_config.ldc = ldc;
    part_config.transA = transA;
    part_config.transB = transB;
    part_config.dtypes = params.dtypes;

    execute_partitioned_matmul(
      layout, trans_input, trans_weight,
      src, weight, dst, bias,
      part_config, params, batch_params,
      is_weights_const, alpha, beta
    );
    // execute_partitioned_matmul() may fall back to aocl_dlp (it sets
    // config.kernel and calls matmul_kernel_wrapper). Propagate the effective
    // kernel back to the caller's reference so matmul_direct()'s post-dispatch
    // guard observes an AOCL-DLP fallback: in an AOCL-DLP-disabled build it
    // then returns status_t::unimplemented instead of reporting success with
    // an uncomputed dst.
    kernel = part_config.kernel;
    return;
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing matmul LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params, bias, kernel);
    return;
  }

#endif

  if (kernel == matmul_algo_t::native_gemm ||
      kernel == matmul_algo_t::native_brgemm) {
    // Native kernels support FP32, BF16, and INT8 (u8/s8 × s8 → f32/bf16/s8/u8)
    const bool is_fp32 = (params.dtypes.src == data_type_t::f32 &&
                          params.dtypes.wei == data_type_t::f32 &&
                          params.dtypes.dst == data_type_t::f32);
    const bool is_bf16 = (params.dtypes.src == data_type_t::bf16 &&
                          params.dtypes.wei == data_type_t::bf16 &&
                          (params.dtypes.dst == data_type_t::bf16 ||
                           params.dtypes.dst == data_type_t::f32));
    const bool is_int8 = ((params.dtypes.src == data_type_t::u8 ||
                           params.dtypes.src == data_type_t::s8) &&
                          params.dtypes.wei == data_type_t::s8 &&
                          (params.dtypes.dst == data_type_t::f32 ||
                           params.dtypes.dst == data_type_t::bf16 ||
                           params.dtypes.dst == data_type_t::s8 ||
                           params.dtypes.dst == data_type_t::u8));
    // TODO: Remove this workaround once the native kernel supports mish
    bool has_mish = false;
    for (size_t i = 0; i < params.postop_.size(); ++i) {
      if (params.postop_[i].po_type == post_op_type_t::mish) {
        has_mish = true;
        break;
      }
    }
    if (!is_fp32 && !is_bf16 && !is_int8) {
      log_info("Native kernel: unsupported data type, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
    else if (transA) {
      log_info("Native kernel: transA not supported, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
    else if (has_mish) {
      log_info("Native kernel: mish post-op not implemented in the native "
               "dispatcher, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    }
    else {
      apilog_info("Executing matmul LOWOHA kernel with ",
                  kernel_to_string(kernel),
                  ", algo: ", static_cast<int>(kernel));
      bool handled = native::native_matmul_execute(kernel, layout, transA, transB,
                     M, N, K, alpha, src, lda,
                     weight, ldb, bias, beta, dst, ldc,
                     is_weights_const, num_threads, params);
      if (handled) {
        return;
      }
      // Native kernel declined (e.g. INT8 shape not supported by GEMV).
      // Fall through to DLP.
      kernel = matmul_algo_t::aocl_dlp;
    }
  }

  apilog_info("Executing matmul LOWOHA kernel without zendnnl-partitioner, algo: ",
              static_cast<int>(kernel));
  matmul_kernel_wrapper(layout, trans_input, trans_weight,
                        M, N, K, alpha,
                        src, lda, weight,
                        ldb, beta, dst, ldc,
                        params.dtypes, kernel,
                        params.mem_format_a, params.mem_format_b, params,
                        batch_params, bias, is_weights_const,
                        num_threads);
}

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       matmul_batch_params_t &batch_params, matmul_params &params) {
  // Profiler overhead in production (ZENDNNL_ENABLE_PROFILER unset):
  //  - profiler_t constructor: eliminated by dead store elimination at -O3
  //    when profiler is never used.
  //  - is_profile_enabled(): negligible (cached static const bool, shared
  //    across all translation units via inline linkage).
  //  - if (is_profile) branch: negligible (always false, branch predictor
  //    learns not-taken quickly).
  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // F16 ISA check must always run — not gated behind diagnostics.
  // Prevents undefined behavior on platforms without AVX512-FP16 support.
  // Covers every F16-bearing operand: src / wei / dst, bias, and any binary
  // post-op buffer. Anything else slips through to a kernel that touches
  // F16 storage and produces wrong results on non-FP16 hardware.
  const bool is_f16 = (params.dtypes.src == data_type_t::f16 ||
                       params.dtypes.wei == data_type_t::f16 ||
                       params.dtypes.dst == data_type_t::f16 ||
                       params.dtypes.bias == data_type_t::f16);
  if (is_f16 && !zendnnl_platform_info().get_avx512_f16_status()) {
    log_error("F16 data type is not supported on this platform "
              "(requires AVX512-FP16).");
    return status_t::isa_unsupported;
  }

  // Validate inputs unless ZENDNNL_DIAGNOSTICS_ENABLE=0 is set (the gate
  // defaults to enabled). In production hot paths where the variable is
  // explicitly disabled, this resolves to a single predicted-taken branch,
  // skipping the full validation path (null-pointer checks, dimension
  // checks, and quantization-parameter validation).
  status_t status = zendnnl::common::op_instrumentation::validate([&]() {
    return validate_matmul_direct_inputs(src, weight, dst, M, N, K,
                                         batch_params.Batch_A, batch_params.Batch_B,
                                         params, is_weights_const);
  });
  if (status != status_t::success) {
    return status;
  }

  // [in,out] Mutates params.postop_[].leading_dim: defaults to N for binary
  // post-ops when the caller leaves it at -1. Must always execute regardless
  // of the ZENDNNL_DIAGNOSTICS_ENABLE flag (i.e. even when diagnostics are
  // explicitly disabled with ZENDNNL_DIAGNOSTICS_ENABLE=0).
  for (auto &po : params.postop_) {
    if (po.po_type == post_op_type_t::binary_add ||
        po.po_type == post_op_type_t::binary_mul) {
      if (po.leading_dim == -1) {
        po.leading_dim = N;
      }
      if (po.dtype == data_type_t::f16 &&
          !zendnnl_platform_info().get_avx512_f16_status()) {
        log_error("F16 binary post-op tensor is not supported on this platform "
                  "(requires AVX512-FP16).");
        return status_t::isa_unsupported;
      }
    }
    else if (po.po_type == post_op_type_t::clip) {
      // matmul_post_op encodes clip bounds as (alpha = lower, beta = upper).
      if (po.alpha > po.beta) {
        log_info("Clip post-op: alpha (", po.alpha,
                 ") > beta (", po.beta, "), swapping to keep lower <= upper.");
        std::swap(po.alpha, po.beta);
      }
    }
  }

  size_t src_type_size = size_of(params.dtypes.src);
  size_t weight_type_size = size_of(params.dtypes.wei);
  size_t out_type_size = size_of(params.dtypes.dst);

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params.num_threads, omp_mt);

  status_t ggml_val_status = zendnnl::common::op_instrumentation::validate([&]() {
    return validate_ggml_packed_inputs(params, is_weights_const,
                                       batch_params.Batch_B, transB);
  });
  if (ggml_val_status != status_t::success) {
    return ggml_val_status;
  }

  // [in,out] Mutates params.dtypes.src, params.quant_params.src_scale.buff,
  // params.quant_params.src_zp.buff, and batch_params.batch_stride_src:
  // dynamic quantization converts the source matrix from its original dtype
  // (f32/bf16) to a lower-precision integer type (s8/u8) in a contiguous
  // buffer, populating scale and zero-point buffers. For batched paths,
  // batch_stride_src is overwritten with the packed stride. This may also
  // change the effective leading dimension (reordered_lda) when the original
  // source has padding (lda > K), since the quantized buffer is packed
  // without padding.
  int reordered_lda = lda;
  reorder_quant_buffers_t quant_buffers;
  if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                                   params, batch_params, transA, M, K,
                                   num_threads, quant_buffers) != status_t::success) {
    return status_t::failure;
  }

  // [in,out] Mutates weight, params.mem_format_b, and
  // params.quant_params.wei_scale: GGML weight unpacking now produces the
  // AOCL-reordered cache entry directly, after source quantization has
  // finalized the dtype and scale metadata used by the reorder path.
  if (params.packing.pack_format_b == 1) {
    if (!ggml_is_sym_quant(params)) {
      log_error("GGML packed weights are supported only for sym-quant "
                "per-group int8 matmul");
      return status_t::failure;
    }
    status_t unpack_status = unpack_ggml_weights_and_cache(weight, N, K, ldb,
                             transB ? 't' : 'n', params);
    if (unpack_status != status_t::success) {
      return unpack_status;
    }
  }

  matmul_algo_t kernel = kernel_select(params, batch_params.Batch_A,
                                       batch_params.Batch_B, batch_count, M,
                                       N, K, num_threads, bias, is_weights_const,
                                       transB);
#if !ZENDNNL_DEPENDS_AOCLDLP
  // AOCL-DLP backed kernels are unavailable in this build. Reject up front
  // with a clear status instead of falling through to the error-out stub.
  if (kernel == matmul_algo_t::aocl_dlp ||
      kernel == matmul_algo_t::aocl_dlp_blocked ||
      kernel == matmul_algo_t::batched_sgemm) {
    log_error("Selected kernel ", kernel_to_string(kernel),
              " requires AOCL-DLP, but ZenDNNL was built without AOCL-DLP "
              "support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    return status_t::unimplemented;
  }
#endif
  matmul_algo_t api_log_kernel = kernel;
  static unsigned int auto_version = get_auto_tuner_ver();

  // TODO: Add memory unreordering logic
  // Unreorder if onednn/ libxsmm is used
  // Implement the necessary logic for memory reordering here
  // if (params.mem_format_b) {}

  // Dispatch to BMM or Matmul based on batch_count
  thread_guard tg(num_threads, omp_mt);
  if (batch_count > 1) {
    // Batch Matrix Multiplication (BMM)
    bmm::bmm_execute(layout, transA, transB,
                     M, N, K, alpha, src,
                     params.dynamic_quant ? reordered_lda : lda,
                     weight, ldb,
                     bias, beta, dst, ldc,
                     is_weights_const, batch_params,
                     src_type_size, weight_type_size, out_type_size, num_threads, kernel, params);
  }
  else {
    // Single Matrix Multiplication (Matmul)
    matmul_execute(layout, transA, transB,
                   M, N, K, alpha, src,
                   params.dynamic_quant ? reordered_lda : lda,
                   weight, ldb,
                   bias, beta, dst, ldc,
                   is_weights_const, src_type_size, out_type_size, num_threads,
                   kernel, params, batch_params, auto_version);
  }

#if !ZENDNNL_DEPENDS_AOCLDLP
  // A native/onednn decline inside matmul_execute can fall back to AOCL-DLP
  // (kernel is taken by reference and mutated, e.g. for an unsupported dtype
  // or transA). In an AOCL-DLP-disabled build that fallback cannot compute,
  // so report the failure to the caller instead of returning success with an
  // uncomputed output buffer.
  if (kernel == matmul_algo_t::aocl_dlp ||
      kernel == matmul_algo_t::aocl_dlp_blocked ||
      kernel == matmul_algo_t::batched_sgemm) {
    log_error("Matmul fell back to AOCL-DLP (", kernel_to_string(kernel),
              "), unavailable in this build (ZENDNNL_DEPENDS_AOCLDLP=0); "
              "output not computed.");
    return status_t::unimplemented;
  }
#endif

  if (is_profile) {
    profiler.tbp_stop();
  }

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA matmul_direct: M=" << M << ", N=" << N << ", K=" << K
       << ", alpha=" << alpha << ", beta=" << beta
       << ", lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc
       << ", transA=" << (transA ? "true" : "false")
       << ", transB=" << (transB ? "true" : "false")
       << ", input_dtype=" << dtype_info(params.dtypes.src)
       << ", weight_dtype=" << dtype_info(params.dtypes.wei)
       << ", output_dtype=" << dtype_info(params.dtypes.dst)
       << ", bias_dtype=" << dtype_info(params.dtypes.bias)
       << ", bias=" << (bias != nullptr ? "true" : "false")
       << ", is_weights_const=" << (is_weights_const ? "true" : "false")
       << ", post_op=[" << post_op_names_to_string(params) << "]"
    << ", post_op_dtype=[" << ([&]() {
      std::string dtypes = post_op_data_types_to_string(params);
      return dtypes.empty() ? "none" : dtypes;
    })()
        << "]"
        << ", Batch_A=" << batch_params.Batch_A << ", Batch_B=" << batch_params.Batch_B
        << ", plugin_op=" << params.plugin_op
        << ", dynamic_quant=" << (params.dynamic_quant ? "true" : "false");

    if (api_log_kernel == matmul_algo_t::auto_tuner) {
      apilog_info(ss.str(), ", kernel=", kernel_to_string(api_log_kernel),
                  ", auto_tuner version=", auto_version);
    }
    else {
      apilog_info(ss.str(), ", kernel=", kernel_to_string(api_log_kernel));
    }
    if (is_profile) {
      profilelog_verbose(ss.str(), ", kernel=", kernel_to_string(kernel),
                         ", weight_address=", static_cast<const void *>(weight),
                         ", time=", profiler.tbp_elapsedtime(), profiler.get_res_str());
    }
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
