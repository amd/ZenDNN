/********************************************************************************
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

#include "matmul_lowoha.hpp"
#include "utils/perf_counters.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace zendnnl {
namespace benchdnn {
namespace matmul {


void set_woq_params(matmul_params &params, const tensor_t &weight_tensor) {
  // Extract weight scale
  const void *scale_buff = weight_tensor.get_quant_scale_raw_handle_const();
  params.quant_params.wei_scale.buff = scale_buff;
  params.quant_params.wei_scale.dt = weight_tensor.get_quant_scale_data_type();
  auto scale_size = weight_tensor.get_quant_scale_size();
  params.quant_params.wei_scale.dims.assign(scale_size.begin(), scale_size.end());
  log_info("LOWOHA WOQ: Weight scale extracted, dims: [",
           params.quant_params.wei_scale.dims.size() > 0 ?
           params.quant_params.wei_scale.dims[0] : 0,
           params.quant_params.wei_scale.dims.size() > 1 ?
           params.quant_params.wei_scale.dims[1] : 0, "]");

  // Extract weight zero point (if asymmetric quantization)
  if (weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
    const void *zp_buff = weight_tensor.get_quant_zero_raw_handle_const();
    if (zp_buff) {
      params.quant_params.wei_zp.buff = zp_buff;
      params.quant_params.wei_zp.dt = weight_tensor.get_quant_zero_data_type();
      auto zp_size = weight_tensor.get_quant_zero_size();
      params.quant_params.wei_zp.dims.assign(zp_size.begin(), zp_size.end());
      log_info("LOWOHA WOQ: Weight zero point extracted");
    }
  }
}

// Pull src scale (and src zp for asymmetric) from a quantized input tensor
// into params. No-op when the tensor is not quantized.
static void set_src_int8_params(matmul_params &params,
                                const tensor_t &input_tensor) {
  if (!input_tensor.is_quantized()) {
    return;
  }
  const void *src_scale_buff = input_tensor.get_quant_scale_raw_handle_const();
  if (src_scale_buff) {
    params.quant_params.src_scale.buff = src_scale_buff;
    params.quant_params.src_scale.dt = input_tensor.get_quant_scale_data_type();
    auto src_scale_size = input_tensor.get_quant_scale_size();
    params.quant_params.src_scale.dims.assign(src_scale_size.begin(),
        src_scale_size.end());
    log_info("LOWOHA INT8: Source scale extracted");
  }
  if (input_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
    const void *src_zp_buff = input_tensor.get_quant_zero_raw_handle_const();
    if (src_zp_buff) {
      params.quant_params.src_zp.buff = src_zp_buff;
      params.quant_params.src_zp.dt = input_tensor.get_quant_zero_data_type();
      auto src_zp_size = input_tensor.get_quant_zero_size();
      params.quant_params.src_zp.dims.assign(src_zp_size.begin(), src_zp_size.end());
      log_info("LOWOHA INT8: Source zero-point extracted");
    }
  }
}

// Pull weight scale (and weight zp for asymmetric) from a quantized weight
// tensor into params. No-op when the tensor is not quantized.
static void set_wei_int8_params(matmul_params &params,
                                const tensor_t &weight_tensor) {
  if (!weight_tensor.is_quantized()) {
    return;
  }
  const void *wei_scale_buff =
    weight_tensor.get_quant_scale_raw_handle_const();
  if (wei_scale_buff) {
    params.quant_params.wei_scale.buff = wei_scale_buff;
    params.quant_params.wei_scale.dt = weight_tensor.get_quant_scale_data_type();
    auto wei_scale_size = weight_tensor.get_quant_scale_size();
    params.quant_params.wei_scale.dims.assign(wei_scale_size.begin(),
        wei_scale_size.end());
    log_info("LOWOHA INT8: Weight scale extracted");
  }
  if (weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
    const void *wei_zp_buff = weight_tensor.get_quant_zero_raw_handle_const();
    if (wei_zp_buff) {
      params.quant_params.wei_zp.buff = wei_zp_buff;
      params.quant_params.wei_zp.dt = weight_tensor.get_quant_zero_data_type();
      auto wei_zp_size = weight_tensor.get_quant_zero_size();
      params.quant_params.wei_zp.dims.assign(wei_zp_size.begin(), wei_zp_size.end());
      log_info("LOWOHA INT8: Weight zero-point extracted");
    }
  }
}

// Pull destination scale (and dst zp for asymmetric) from a quantized output
// tensor into params. No-op when the tensor is not quantized.
static void set_dst_int8_params(matmul_params &params,
                                const tensor_t &output_tensor) {
  if (!output_tensor.is_quantized()) {
    return;
  }
  const void *dst_scale_buff =
    output_tensor.get_quant_scale_raw_handle_const();
  if (dst_scale_buff) {
    params.quant_params.dst_scale.buff = dst_scale_buff;
    params.quant_params.dst_scale.dt = output_tensor.get_quant_scale_data_type();
    auto dst_scale_size = output_tensor.get_quant_scale_size();
    params.quant_params.dst_scale.dims.assign(dst_scale_size.begin(),
        dst_scale_size.end());
    log_info("LOWOHA INT8: Destination scale extracted");
  }
  if (output_tensor.get_quant_subtype() == quant_subtype_t::asymmetric) {
    const void *dst_zp_buff = output_tensor.get_quant_zero_raw_handle_const();
    if (dst_zp_buff) {
      params.quant_params.dst_zp.buff = dst_zp_buff;
      params.quant_params.dst_zp.dt = output_tensor.get_quant_zero_data_type();
      auto dst_zp_size = output_tensor.get_quant_zero_size();
      params.quant_params.dst_zp.dims.assign(dst_zp_size.begin(), dst_zp_size.end());
      log_info("LOWOHA INT8: Destination zero-point extracted");
    }
  }
}

void set_int8_params(matmul_params &params, const tensor_t &input_tensor,
                     const tensor_t &weight_tensor, const tensor_t &output_tensor) {
  set_src_int8_params(params, input_tensor);
  set_wei_int8_params(params, weight_tensor);
  set_dst_int8_params(params, output_tensor);
}

// Configure dynamic source quantization for W8A8 and W4A8 paths. The src
// tensor must already carry a properly-shaped scale tensor (created by
// create_input_tensor); the runtime fills the scale buffer during the
// dynamic-quant pass.
void set_dyn_quant_src_params(matmul_params &params,
                              const tensor_t &input_tensor,
                              const tensor_t &weight_tensor) {
  params.dynamic_quant = true;
  params.dtypes.compute = data_type_t::s8;
  set_src_int8_params(params, input_tensor);
  set_wei_int8_params(params, weight_tensor);
  log_info("LOWOHA dyn-quant: enabled (src=",
           datatypeToStr(input_tensor.get_data_type()),
           ", wei=", datatypeToStr(weight_tensor.get_data_type()),
           ", compute=s8)");
}

// W8A8: src in {bf16,f32} + wei=s8
// W4A8: bf16 src/dst + wei=s4
static bool is_dynamic_quant_dtype(const tensor_t &input_tensor,
                                   const tensor_t &weight_tensor,
                                   const tensor_t &output_tensor) {
  const auto src_dt = input_tensor.get_data_type();
  const auto wei_dt = weight_tensor.get_data_type();
  const auto dst_dt = output_tensor.get_data_type();
  bool w8a8 = wei_dt == data_type_t::s8 &&
              (src_dt == data_type_t::bf16 || src_dt == data_type_t::f32);
  bool w4a8 = wei_dt == data_type_t::s4 &&
              src_dt == data_type_t::bf16 &&
              dst_dt == data_type_t::bf16;
  return w8a8 || w4a8;
}

void set_lowoha_matmul_params(matmul_params &params, int &lda, int &ldb,
                              int &ldc,
                              const tensor_t &input_tensor,
                              const tensor_t &weight_tensor, const tensor_t &output_tensor,
                              const tensor_t &bias, const std::vector<tensor_t> &binary_post_ops_tensors,
                              const MatmulConfig &cfg) {

  auto input_dim              = input_tensor.get_dim();
  auto weight_dim             = weight_tensor.get_dim();
  auto output_dim             = output_tensor.get_dim();

  lda             = cfg.isTransA ?
                    input_tensor.get_stride(input_dim-1) :
                    input_tensor.get_stride(input_dim-2);
  ldb             = cfg.isTransB ?
                    weight_tensor.get_stride(weight_dim-1):
                    weight_tensor.get_stride(weight_dim-2);
  ldc             = output_tensor.get_stride(output_dim-2);

  matmul_data_types matmul_dtypes;
  matmul_dtypes.src = input_tensor.get_data_type();
  matmul_dtypes.wei = weight_tensor.get_data_type();
  matmul_dtypes.dst = output_tensor.get_data_type();
  if (cfg.isBiasEnabled) {
    matmul_dtypes.bias = bias.get_data_type();
  }
  matmul_dtypes.compute = data_type_t::none;
  params.dtypes = matmul_dtypes;

  // Check if this is INT8 quantization
  bool is_int8 = (input_tensor.get_data_type() == data_type_t::u8 ||
                  input_tensor.get_data_type() == data_type_t::s8) &&
                 weight_tensor.get_data_type() == data_type_t::s8;

  // Dynamic source quantization (W8A8 / W4A8).
  // Mirrors the runtime gate in reorder_quantization.cpp. Mutually exclusive
  // with the static int8 (s8/u8 src) path above. We additionally require the
  // source tensor to (a) be 2D and (b) carry an attached src-scale buffer.
  // Pipeline layers (i > 0) feed the previous layer's bf16 output as input;
  // those tensors have no src-scale, so without this gate
  // set_dyn_quant_src_params() would leave params.quant_params.src_scale.buff
  // unset and crash at runtime.
  const bool dyn_quant_src_dtype_match =
    is_dynamic_quant_dtype(input_tensor, weight_tensor, output_tensor);
  const bool dyn_quant_src_is_2d = input_tensor.get_dim() == 2;
  // Guard the scale-buffer probe with is_quantized() first; otherwise the
  // tensor API logs an apilog_error every time we ask a non-quantized input
  // (e.g. plain bf16/f32 src) for its scale handle.
  const bool dyn_quant_src_has_scale =
    input_tensor.is_quantized() &&
    input_tensor.get_quant_scale_raw_handle_const() != nullptr;
  bool is_dyn_quant_src =
    cfg.src_dynamic_quant &&
    dyn_quant_src_dtype_match &&
    dyn_quant_src_is_2d &&
    dyn_quant_src_has_scale;

  // WOQ (Weight-Only Quantization): BF16 src + S4/U4 weights without W4A8
  // dynamic source quantization.
  bool is_woq = (input_tensor.get_data_type() == data_type_t::bf16 &&
                 (weight_tensor.get_data_type() == data_type_t::s4 ||
                  weight_tensor.get_data_type() == data_type_t::u4) &&
                 !is_dyn_quant_src);

  if (cfg.src_dynamic_quant && dyn_quant_src_dtype_match && !is_dyn_quant_src) {
    commonlog_warning(
      "LOWOHA: disabling dynamic source quantization for this layer; "
      "source tensor is missing quant-scale metadata or has unsupported "
      "shape (ndims=", input_tensor.get_dim(),
      ", has_src_scale=", dyn_quant_src_has_scale ? 1 : 0, ").");
  }

  if (is_woq) {
    set_woq_params(params, weight_tensor);
  }

  // For INT8: Extract quantization parameters from all tensors
  if (is_int8) {
    set_int8_params(params, input_tensor, weight_tensor, output_tensor);
  }

  if (is_dyn_quant_src) {
    set_dyn_quant_src_params(params, input_tensor, weight_tensor);
  }

  const auto &binary_post_op = binary_post_ops_tensors;
  for (auto k = 0; k < cfg.post_ops.size(); k++) {
    zendnnl::lowoha::matmul::matmul_post_op postop_item;
    postop_item.po_type = cfg.post_ops[k];
    for (auto l = 0; l < binary_post_op.size(); l++) {
      if (k == cfg.binary_post_ops_pos[l]) {
        postop_item.buff = binary_post_op[l].get_raw_handle_unsafe();
        postop_item.dtype = binary_post_op[l].get_data_type();
        auto binary_tensor_dims = binary_post_op[l].get_size();
        postop_item.dims.assign(binary_tensor_dims.begin(), binary_tensor_dims.end());
        break;
      }
    }
    if (postop_item.buff == nullptr) {
      postop_item.dtype = output_tensor.get_data_type();
    }
    if (cfg.post_ops[k] == zendnnl::ops::post_op_type_t::clip) {
      postop_item.alpha = -0.5f;
      postop_item.beta  =  0.5f;
    }
    params.postop_.push_back(postop_item);
  }
}

int matmul_lowoha_benchdnn(std::vector<MatmulConfig> configs,
                           std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                           &matmul_results, const global_options &options,
                           size_t cache_size) {

  bool skip;
  for (auto &cfg:configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t input_tensor;
      int num_weight_buffers = 1;
      std::vector<std::vector<tensor_t>> weights_buffer_pool(num_weight_buffers);
      std::vector<tensor_t> bias, output_tensor;
      std::vector<std::vector<tensor_t>> binary_post_ops_tensors;

      // true indicates LOWOHA mode for weight tensor creation
      int ret = create_weights_tensor(tensor_factory, cfg, weights_buffer_pool[0],
                                      options,
                                      true);
      if (ret != OK) {
        testlog_error("create_weights_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      if (options.cache_mode == CacheMode::WARM) {
        // Warm cache: rotate several weight buffers across iterations so one resident
        // copy does not dominate timings. Auto (-1): size the pool from weight bytes vs
        // CACHE_SIZE_MULTIPLIER * cache_size, then clamp with MIN_NUM_WEIGHT_BUFFERS and
        // the iteration budget (cfg.iters when set, else MIN_NUM_WEIGHT_BUFFERS).
        if (options.num_weight_buffers == -1) {
          int num_weight_buffers_ = (cfg.iters > 0) ? cfg.iters : MIN_NUM_WEIGHT_BUFFERS;
          size_t weight_bytes = 0;
          for (const auto &w : weights_buffer_pool[0]) {
            weight_bytes += w.get_buffer_sz_bytes();
          }
          auto cache_size_ = cache_size * CACHE_SIZE_MULTIPLIER;
          if (weight_bytes > 0 && cache_size_ > 0) {
            // ceil(cache_size_ / weight_bytes) + 1 so total weight bytes exceed scaled cache.
            num_weight_buffers = (cache_size_ + weight_bytes - 1) / weight_bytes + 1;
            num_weight_buffers = std::max(num_weight_buffers, MIN_NUM_WEIGHT_BUFFERS);
          }
          else {
            num_weight_buffers = num_weight_buffers_;
          }
          num_weight_buffers = std::min(num_weight_buffers, num_weight_buffers_);
        }
        else {
          // User-specified count; cannot use more buffers than benchmark iterations.
          if (cfg.iters > 0 && options.num_weight_buffers > cfg.iters) {
            num_weight_buffers = cfg.iters;
            commonlog_warning("num_weight_buffers (", options.num_weight_buffers,
                              ") exceeds iters (", cfg.iters, "); using ", num_weight_buffers);
          }
          else {
            num_weight_buffers = options.num_weight_buffers;
          }
        }

        weights_buffer_pool.resize(num_weight_buffers);
        for (size_t j = 1; j < weights_buffer_pool.size(); j++) {
          ret = create_weights_tensor(tensor_factory, cfg, weights_buffer_pool[j],
                                      options, true);
          if (ret != OK) {
            testlog_error("create_weights_tensor failed");
            log_benchmark_failure(cfg);
            skip = true;
            break;
          }
        }
        if (skip) {
          continue;
        }
      }

      ret = create_bias_tensor(tensor_factory, cfg, bias, options);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_input_tensor(tensor_factory, cfg, input_tensor, options,
                                /* isLOWOHA = */ true);
      if (ret != OK) {
        testlog_error("create_input_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_output_tensor(tensor_factory, cfg, output_tensor, options);
      if (ret != OK) {
        testlog_error("create_output_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_binary_post_ops_tensors(tensor_factory, cfg,
                                           binary_post_ops_tensors);
      if (ret != OK) {
        testlog_error("create_binary_post_ops_tensors failed");
        log_benchmark_failure(cfg);
        continue;
      }

      float alpha = cfg.alpha, beta = cfg.beta;
      bool is_weights_const       = cfg.is_weights_const;

      const int batchA = cfg.bs;
      const int batchB = cfg.bs;
      const int M = cfg.m;

      matmul_batch_params_t batch_params;
      batch_params.Batch_A = batchA;
      batch_params.Batch_B = batchB;

      TimingStats time_stats;

      PerfProfile perf_profile = parse_perf_profile(
                                   options.perf_profile_str.c_str());
      PerfCounterGroup perf_ctrs(perf_profile);
      const bool use_perf = options.perf_counters && perf_ctrs.is_available();
      if (use_perf) {
        if (!perf_ctrs.open()) {
          commonlog_warning("HW perf counters unavailable; continuing without them.");
        }
      }

      if (options.cache_mode == CacheMode::WARM) {
        flush_cache(cache_size);
      }
      // warm-up iterations
      for (auto j = 0; j < cfg.warmup_iters && !skip; j++) {
        for (auto i = 0; i < cfg.n_values.size(); i++) {

          const int K = (i == 0) ? cfg.k : cfg.n_values[i - 1];
          const int N = cfg.n_values[i];
          auto input_tensor_ = (i == 0) ? input_tensor : output_tensor[i - 1];
          auto weight_tensor_ = weights_buffer_pool[0][i];
          auto output_tensor_ = output_tensor[i];
          int lda, ldb, ldc;
          matmul_params params;
          set_lowoha_matmul_params(params, lda, ldb, ldc, input_tensor_, weight_tensor_,
                                   output_tensor_, bias[i], binary_post_ops_tensors[i], cfg);

          void *A_data = input_tensor_.get_raw_handle_unsafe();
          void *B_data = weight_tensor_.get_raw_handle_unsafe();
          void *C_data = output_tensor_.get_raw_handle_unsafe();
          void *bias_data = nullptr;
          if (cfg.isBiasEnabled) {
            bias_data = bias[i].get_raw_handle_unsafe();
          }

          status_t status = matmul_direct(
                              'r',  // layout: row-major
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, bias_data,
                              beta, C_data, ldc, is_weights_const,
                              batch_params, params);
          if (status != status_t::success) {
            testlog_error("LOWOHA: Matmul execution failed.");
            skip = true;
            break;
          }
        }
        if (skip) {
          log_benchmark_failure(cfg);
          break;
        }
      }
      if (skip) {
        continue;
      }

      std::vector<TimingStats> time_stats_layer(cfg.n_values.size());
      std::vector<double> elapsed_ms_layer(cfg.n_values.size(), 0.0);

      if (use_perf && perf_ctrs.is_available()) {
        perf_ctrs.reset();
        perf_ctrs.enable();
      }

      for (auto j = 0; j < cfg.iters && !skip; j++) {
        if (options.cache_mode == CacheMode::COLD) {
          flush_cache(cache_size);
        }
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          const int K = (i == 0) ? cfg.k : cfg.n_values[i - 1];
          const int N = cfg.n_values[i];
          auto input_tensor_ = (i == 0) ? input_tensor : output_tensor[i - 1];
          auto weight_tensor_ = (options.cache_mode == CacheMode::WARM) ?
                                weights_buffer_pool[(1 + j) % num_weight_buffers][i] :
                                weights_buffer_pool[0][i];
          auto output_tensor_ = output_tensor[i];
          auto input_dim              = input_tensor_.get_dim();
          auto weight_dim             = weight_tensor_.get_dim();
          auto output_dim             = output_tensor_.get_dim();

          int lda, ldb, ldc;
          matmul_params params;
          set_lowoha_matmul_params(params, lda, ldb, ldc, input_tensor_, weight_tensor_,
                                   output_tensor_, bias[i], binary_post_ops_tensors[i], cfg);

          void *A_data = input_tensor_.get_raw_handle_unsafe();
          void *B_data = weight_tensor_.get_raw_handle_unsafe();
          void *C_data = output_tensor_.get_raw_handle_unsafe();
          void *bias_data = nullptr;
          if (cfg.isBiasEnabled) {
            bias_data = bias[i].get_raw_handle_unsafe();
          }

          auto start_layer = std::chrono::high_resolution_clock::now();
          TimingStats time_stats;
          status_t status = matmul_direct(
                              'r',
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, bias_data,
                              beta, C_data, ldc, is_weights_const,
                              batch_params, params);
          if (status != status_t::success) {
            testlog_error("LOWOHA: Matmul execution failed.");
            skip = true;
            break;
          }

          auto end_layer = std::chrono::high_resolution_clock::now();
          double time_taken = (std::chrono::duration<double, std::milli>
                               (end_layer - start_layer).count());
          elapsed_ms_layer[i] += time_taken;
        }
        if (skip) {
          log_benchmark_failure(cfg);
          break;
        }
      }

      if (use_perf && perf_ctrs.is_available()) {
        perf_ctrs.disable();
        perf_ctrs.read();
      }

      if (skip) {
        continue;
      }

      for (size_t i = 0; i < cfg.n_values.size(); i++) {
        time_stats_layer[i].total_time_ms = elapsed_ms_layer[i];
      }
      print_matmul_execution_summary(cfg, time_stats_layer, options);

      if (use_perf && perf_ctrs.is_available()) {
        double elapsed_sec = 0;
        for (auto &ts : time_stats_layer) {
          elapsed_sec += ts.total_time_ms / 1000.0;
        }
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        if (nt < 1) {
          nt = 1;
        }
        auto derived = perf_ctrs.derive(elapsed_sec, nt);
        printf("  [PERF]");
        perf_ctrs.print_values(derived, false);
        printf("\n");
        perf_ctrs.print_raw_counters();
      }
      matmul_results.emplace_back(cfg, time_stats_layer);
    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  if (skip) {
    return NOT_OK;
  }
  return OK;
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl