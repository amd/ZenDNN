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
#include "partitioning/bmm/bmm_partitioner.hpp"
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
                           const void *bias, bool is_weights_const) {
#if ZENDNNL_DEPENDS_LIBXSMM
  if (kernel == matmul_algo_t::libxsmm) {
    log_info("Using libxsmm kernel");
    if (run_libxsmm(transA, transB, M, N, K, beta, lda, ldb, ldc, A, B, C, dtypes,
                    lowoha_param, bias)) {
      return;
    }
  }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    log_info("Using onednn kernel");
    matmul_onednn_wrapper(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                          ldc, lowoha_param, batch_params, bias, kernel, is_weights_const);
    return;
  }
#endif
  log_info("Using AOCL DLP kernel");
  run_dlp(layout, transA, transB, M, N, K, alpha, beta,
          lda, ldb, ldc, mem_format_a, mem_format_b,
          A, B, C, dtypes, lowoha_param, bias, kernel, is_weights_const);
  return;
}

void bmm_execute(const char layout, const bool transA, const bool transB,
                 const int M, const int N, const int K, const float alpha,
                 const void *src, const int lda,
                 const void *weight, const int ldb,
                 const void *bias, const float beta,
                 void *dst, const int ldc,
                 const bool is_weights_const,
                 matmul_batch_params_t &batch_params,
                 const size_t src_type_size, const size_t out_type_size,
                 const int num_threads,
                 matmul_algo_t &kernel, matmul_params &params) {

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const char trans_input  = transA ? 't' : 'n';
  const char trans_weight = transB ? 't' : 'n';

  // Calculate batch strides in elements (not bytes)
  size_t src_batch_stride_elems = (batch_params.batch_stride_src !=
                                   static_cast<size_t>(-1))
                                  ? batch_params.batch_stride_src
                                  : (transA ? K *lda : M * lda);
  size_t weight_batch_stride_elems = (batch_params.batch_stride_wei !=
                                      static_cast<size_t>(-1))
                                     ? batch_params.batch_stride_wei
                                     : (transB ? N *ldb : K * ldb);
  size_t dst_batch_stride_elems = (batch_params.batch_stride_dst !=
                                   static_cast<size_t>(-1))
                                  ? batch_params.batch_stride_dst
                                  : M * ldc;

  size_t src_batch_stride_bytes = src_batch_stride_elems * src_type_size;
  size_t weight_batch_stride_bytes = weight_batch_stride_elems * src_type_size;
  size_t dst_batch_stride_bytes = dst_batch_stride_elems * out_type_size;

  if (kernel == matmul_algo_t::batched_sgemm) {
    apilog_info("Executing BMM LOWOHA kernel with batch SGEMM, algo: ",
                static_cast<int>(kernel));

    matmul_batch_gemm_wrapper(layout, trans_input, trans_weight,
                              M, N, K, alpha,
                              src, lda, weight, ldb, beta, dst, ldc,
                              params.dtypes, batch_count,
                              batch_params.Batch_A, batch_params.Batch_B,
                              params.mem_format_a, params.mem_format_b,
                              src_batch_stride_bytes, weight_batch_stride_bytes, dst_batch_stride_bytes,
                              params, bias, num_threads);
    return;
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing BMM LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params, bias, kernel,
                          is_weights_const, src_batch_stride_elems, weight_batch_stride_elems,
                          dst_batch_stride_elems);
    return;
  }
#endif

  if (num_threads > 1) {
    // Setup partition configuration
    bmm_partition_config_t part_config;
    part_config.M = M;
    part_config.N = N;
    part_config.K = K;
    part_config.batch_count = batch_count;
    part_config.num_threads = num_threads;
    part_config.kernel = kernel;
    part_config.src_batch_stride_bytes = src_batch_stride_bytes;
    part_config.weight_batch_stride_bytes = weight_batch_stride_bytes;
    part_config.dst_batch_stride_bytes = dst_batch_stride_bytes;

    // Execute partitioned BMM with all logic encapsulated
    execute_bmm_partition(
      src, weight, dst, bias,
      part_config, batch_params, params,
      layout, trans_input, trans_weight, transA,
      alpha, beta, lda, ldb, ldc,
      src_type_size, out_type_size, is_weights_const);
  }
  else {
    // Single thread execution for batches
    if (kernel == matmul_algo_t::libxsmm &&
        !(can_use_libxsmm(trans_input, trans_weight, M, N, K, alpha, beta,
                          params))) {
      kernel = matmul_algo_t::aocl_dlp;
    }

    apilog_info("Executing BMM LOWOHA kernel without zendnnl-partitioner, algo: ",
                static_cast<int>(kernel));

    for (int b = 0; b < batch_count; ++b) {
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) +
                               get_batch_index(b, batch_params.Batch_A) * src_batch_stride_bytes;
      const uint8_t *weight_ptr = static_cast<const uint8_t *>(weight) +
                                  get_batch_index(b, batch_params.Batch_B) * weight_batch_stride_bytes;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + b * dst_batch_stride_bytes;

      // Create batch-specific params with offset post-op buffers for 3D post-ops
      matmul_params batch_lowoha_params = params;
      apply_bmm_postop_offsets(batch_lowoha_params, b, 0, N);

      matmul_kernel_wrapper(layout, trans_input, trans_weight,
                            M, N, K, alpha,
                            src_ptr, lda, weight_ptr,
                            ldb, beta, dst_ptr, ldc,
                            params.dtypes, kernel,
                            params.mem_format_a, params.mem_format_b, batch_lowoha_params,
                            batch_params, bias, is_weights_const);
    }
  }
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

  // Auto-tuner kernel selection for single batch
  if (kernel == matmul_algo_t::auto_tuner) {
    if (auto_version == 1) {
      kernel = auto_compute_matmul_v1(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const);
    }
    else {
      kernel = auto_compute_matmul_v2(layout, trans_input, trans_weight, M,
                                      N, K, alpha, src, lda, weight, ldb,
                                      beta, dst, ldc, params.dtypes,
                                      kernel, params.mem_format_a, params.mem_format_b,
                                      params, batch_params, bias, is_weights_const);
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
    return;
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (kernel == matmul_algo_t::onednn ||
      kernel == matmul_algo_t::onednn_blocked) {
    apilog_info("Executing matmul LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

    matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src, lda,
                          weight, ldb, beta, dst, ldc, params, batch_params, bias, kernel,
                          is_weights_const);
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
    if (!is_fp32 && !is_bf16 && !is_int8) {
      log_info("Native kernel: unsupported data type, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    } else if (transA) {
      log_info("Native kernel: transA not supported, falling back to aocl_dlp");
      kernel = matmul_algo_t::aocl_dlp;
    } else {
      apilog_info("Executing matmul LOWOHA kernel with ",
                  kernel_to_string(kernel),
                  ", algo: ", static_cast<int>(kernel));
      bool handled = native::native_matmul_execute(kernel, layout, transA, transB,
                              M, N, K, alpha, src, lda,
                              weight, ldb, bias, beta, dst, ldc,
                              is_weights_const, num_threads, params);
      if (handled) return;
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
                        batch_params, bias, is_weights_const);
}

status_t matmul_direct(const char layout, const bool transA, const bool transB,
                       const int M, const int N, const int K, const float alpha, const void *src,
                       const int lda, const void *weight, const int ldb, const void *bias,
                       const float beta, void *dst, const int ldc, const bool is_weights_const,
                       matmul_batch_params_t batch_params, matmul_params params) {
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
  const bool is_f16 = (params.dtypes.src == data_type_t::f16 ||
                       params.dtypes.wei == data_type_t::f16 ||
                       params.dtypes.dst == data_type_t::f16);
  if (is_f16 && !zendnnl_platform_info().get_f16_status()) {
    log_error("F16 data type is not supported on this platform "
              "(requires AVX512-FP16 or AVX-NE-CONVERT ISA).");
    return status_t::isa_unsupported;
  }

  // Validate inputs only when ZENDNNL_DIAGNOSTICS_ENABLE=1. In production this
  // resolves to a single predicted-not-taken branch, skipping the full
  // validation path (null-pointer checks, dimension checks, and
  // quantization-parameter validation).
  status_t status = zendnnl::common::op_instrumentation::validate([&]() {
    return validate_matmul_direct_inputs(src, weight, dst, M, N, K,
                    batch_params.Batch_A, batch_params.Batch_B,
                    params, is_weights_const);
  });
  if (status != status_t::success) {
    return status;
  }

  // Set leading dimension for binary post-op buffers if not set.
  // This is parameter initialization, not validation — must always execute
  // regardless of the ZENDNNL_DIAGNOSTICS_ENABLE flag.
  for (auto &po : params.postop_) {
    if (po.po_type == post_op_type_t::binary_add ||
        po.po_type == post_op_type_t::binary_mul) {
      if (po.leading_dim == -1) {
        po.leading_dim = N;
      }
    }
  }

  if (params.packing.pack_format_b == 1) {
    status_t unpack_status = unpack_ggml_weights_and_cache(weight, N, K, params);
    if (unpack_status != status_t::success) {
      return unpack_status;
    }
  }

  size_t src_type_size = size_of(params.dtypes.src);
  size_t out_type_size = size_of(params.dtypes.dst);

  const int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params.num_threads, omp_mt);

  // Dynamic quantization converts the source matrix from its original dtype
  // (f32/bf16) to a lower-precision integer type (s8/u8) in a contiguous buffer.
  // This may change the effective leading dimension (reordered_lda) when the
  // original source has padding (lda > K), since the quantized buffer is packed
  // without padding. Non-quantized paths must continue using the original lda.
  int reordered_lda = lda;
  reorder_quant_buffers_t quant_buffers;
  if (reorder_quantization_wrapper(src, lda, reordered_lda, src_type_size,
                                   params, batch_params, transA, M, K,
                                   num_threads, quant_buffers) != status_t::success) {
    return status_t::failure;
  }

  matmul_algo_t kernel = kernel_select(params, batch_params.Batch_A,
                                       batch_params.Batch_B, batch_count, M,
                                       N, K, num_threads, bias, is_weights_const);
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
    bmm_execute(layout, transA, transB,
                M, N, K, alpha, src,
                params.dynamic_quant ? reordered_lda : lda,
                weight, ldb,
                bias, beta, dst, ldc,
                is_weights_const, batch_params,
                src_type_size, out_type_size, num_threads, kernel, params);
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
