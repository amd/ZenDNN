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

namespace zendnnl {
namespace benchdnn {
namespace matmul {


// TODO:
// - Add pipeline (multi-layer) support.
int matmul_lowoha_benchdnn(std::vector<MatmulConfig> configs,
                           std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                           &matmul_results, const global_options &options,
                           size_t cache_size) {

  bool skip;
  for (const auto &cfg:configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t input_tensor;
      std::vector<tensor_t> weight_tensor, bias, output_tensor;
      std::vector<std::vector<tensor_t>> binary_post_ops_tensors;

      // true indicates LOWOHA mode for weight tensor creation
      int ret = create_weights_tensor(tensor_factory, cfg, weight_tensor, options,
                                      true);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_bias_tensor(tensor_factory, cfg, bias, options);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_input_tensor(tensor_factory, cfg, input_tensor, options);
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
      auto input_dim              = input_tensor.get_dim();
      auto weight_dim             = weight_tensor[0].get_dim();
      auto output_dim             = output_tensor[0].get_dim();

      const int   lda             = cfg.isTransA ?
                                    input_tensor.get_stride(input_dim-1) :
                                    input_tensor.get_stride(input_dim-2);
      const int   ldb             = cfg.isTransB ?
                                    weight_tensor[0].get_stride(weight_dim-1):
                                    weight_tensor[0].get_stride(weight_dim-2);
      const int   ldc             = output_tensor[0].get_stride(output_dim-2);

      const int batchA = cfg.bs;
      const int batchB = cfg.bs;
      const int M = cfg.m;
      const int K = cfg.k;
      const int N = cfg.n_values[0];

      // Validate dimensions
      if (M == 0 || K == 0 || N == 0) {
        testlog_error("LOWOHA: Invalid tensor dimensions - M:", M, " K:", K, " N:", N);
        log_benchmark_failure(cfg);
        continue;
      }

      matmul_data_types matmul_dtypes;
      matmul_dtypes.src = cfg.dt[0];
      matmul_dtypes.wei = cfg.dt[1];
      matmul_dtypes.dst = cfg.dt[2];
      matmul_dtypes.bias = cfg.bias_dt;
      matmul_dtypes.compute = data_type_t::none;

      matmul_params params;
      params.dtypes = matmul_dtypes;

      matmul_batch_params_t batch_params;
      batch_params.Batch_A = batchA;
      batch_params.Batch_B = batchB;

      // Validate data types
      if (cfg.dt[0] != data_type_t::f32 && cfg.dt[0] != data_type_t::bf16 &&
          cfg.dt[0] != data_type_t::u8 && cfg.dt[0] != data_type_t::s8) {
        testlog_error("LOWOHA: Unsupported source data type");
        log_benchmark_failure(cfg);
        continue;
      }
      if (cfg.dt[1] != data_type_t::f32 && cfg.dt[1] != data_type_t::bf16 &&
          cfg.dt[1] != data_type_t::u8 && cfg.dt[1] != data_type_t::s8) {
        testlog_error("LOWOHA: Unsupported weight data type");
        log_benchmark_failure(cfg);
        continue;
      }
      if (cfg.dt[2] != data_type_t::f32 && cfg.dt[2] != data_type_t::bf16 &&
          cfg.dt[2] != data_type_t::u8 && cfg.dt[2] != data_type_t::s8 &&
          cfg.dt[2] != data_type_t::s32) {
        testlog_error("LOWOHA: Unsupported output data type");
        log_benchmark_failure(cfg);
        continue;
      }

      // Check if this is INT8 quantization
      bool is_int8 = (cfg.dt[0] == data_type_t::u8 ||
                      cfg.dt[0] == data_type_t::s8) &&
                     cfg.dt[1] == data_type_t::s8;

      // For INT8: Extract quantization parameters from all tensors
      if (is_int8) {
        // Extract source scale
        if (input_tensor.is_quantized()) {
          const void *src_scale_buff = input_tensor.get_quant_scale_raw_handle_const();
          if (src_scale_buff) {
            params.quant_params.src_scale.buff = src_scale_buff;
            params.quant_params.src_scale.dt = input_tensor.get_quant_scale_data_type();
            auto src_scale_size = input_tensor.get_quant_scale_size();
            params.quant_params.src_scale.dims.assign(src_scale_size.begin(),
                src_scale_size.end());
            log_info("LOWOHA INT8: Source scale extracted");
          }
          // Extract source zero point (for asymmetric quantization)
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

        // Extract weight scale
        if (weight_tensor[0].is_quantized()) {
          const void *wei_scale_buff =
            weight_tensor[0].get_quant_scale_raw_handle_const();
          if (wei_scale_buff) {
            params.quant_params.wei_scale.buff = wei_scale_buff;
            params.quant_params.wei_scale.dt = weight_tensor[0].get_quant_scale_data_type();
            auto wei_scale_size = weight_tensor[0].get_quant_scale_size();
            params.quant_params.wei_scale.dims.assign(wei_scale_size.begin(),
                wei_scale_size.end());
            log_info("LOWOHA INT8: Weight scale extracted");
          }
          // Extract weight zero point (for asymmetric quantization)
          if (weight_tensor[0].get_quant_subtype() == quant_subtype_t::asymmetric) {
            const void *wei_zp_buff = weight_tensor[0].get_quant_zero_raw_handle_const();
            if (wei_zp_buff) {
              params.quant_params.wei_zp.buff = wei_zp_buff;
              params.quant_params.wei_zp.dt = weight_tensor[0].get_quant_zero_data_type();
              auto wei_zp_size = weight_tensor[0].get_quant_zero_size();
              params.quant_params.wei_zp.dims.assign(wei_zp_size.begin(), wei_zp_size.end());
              log_info("LOWOHA INT8: Weight zero-point extracted");
            }
          }
        }

        // Extract destination scale and zero-point
        if (output_tensor[0].is_quantized()) {
          const void *dst_scale_buff =
            output_tensor[0].get_quant_scale_raw_handle_const();
          if (dst_scale_buff) {
            params.quant_params.dst_scale.buff = dst_scale_buff;
            params.quant_params.dst_scale.dt = output_tensor[0].get_quant_scale_data_type();
            auto dst_scale_size = output_tensor[0].get_quant_scale_size();
            params.quant_params.dst_scale.dims.assign(dst_scale_size.begin(),
                dst_scale_size.end());
            log_info("LOWOHA INT8: Destination scale extracted");
          }
          // Extract destination zero point (for asymmetric quantization)
          if (output_tensor[0].get_quant_subtype() == quant_subtype_t::asymmetric) {
            const void *dst_zp_buff = output_tensor[0].get_quant_zero_raw_handle_const();
            if (dst_zp_buff) {
              params.quant_params.dst_zp.buff = dst_zp_buff;
              params.quant_params.dst_zp.dt = output_tensor[0].get_quant_zero_data_type();
              auto dst_zp_size = output_tensor[0].get_quant_zero_size();
              params.quant_params.dst_zp.dims.assign(dst_zp_size.begin(), dst_zp_size.end());
              log_info("LOWOHA INT8: Destination zero-point extracted");
            }
          }
        }
      }

      for (auto k = 0; k < cfg.n_values.size(); k++) {
        const auto &binary_post_op = binary_post_ops_tensors[k];
        for (auto j = 0; j < cfg.post_ops.size(); j++) {
          zendnnl::lowoha::matmul::matmul_post_op postop_item;
          postop_item.po_type = cfg.post_ops[j];
          for (auto i = 0; i < binary_post_op.size(); i++) {
            if (j == cfg.binary_post_ops_pos[i]) {
              postop_item.buff = binary_post_op[i].get_raw_handle_unsafe();
              postop_item.dtype = binary_post_op[i].get_data_type();
              auto binary_tensor_dims = binary_post_op[i].get_size();
              postop_item.dims.assign(binary_tensor_dims.begin(), binary_tensor_dims.end());
              break;
            }
          }
          params.postop_.push_back(postop_item);
        }
      }

      TimingStats time_stats;
      // warm-up iterations
      for (auto j = 0; j < cfg.warmup_iters && !skip; j++) {
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          void *A_data = (i == 0) ? input_tensor.get_raw_handle_unsafe() :
                         output_tensor[i - 1].get_raw_handle_unsafe();
          void *B_data = weight_tensor[i].get_raw_handle_unsafe();
          void *C_data = output_tensor[i].get_raw_handle_unsafe();
          void *bias_data = nullptr;
          if (cfg.isBiasEnabled) {
            bias_data = bias[i].get_raw_handle_unsafe();
          }

          status_t status = matmul_direct(
                              'r',  // layout: row-major
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, bias_data,
                              beta, C_data, ldc, true,
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

      for (auto j = 0; j < cfg.iters && !skip; j++) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          auto start_layer = std::chrono::high_resolution_clock::now();
          TimingStats time_stats; // Per-layer, per-iteration

          void *A_data = (i == 0) ? input_tensor.get_raw_handle_unsafe() :
                         output_tensor[i - 1].get_raw_handle_unsafe();
          void *B_data = weight_tensor[i].get_raw_handle_unsafe();
          void *C_data = output_tensor[i].get_raw_handle_unsafe();
          void *bias_data = nullptr;
          if (cfg.isBiasEnabled) {
            bias_data = bias[i].get_raw_handle_unsafe();
          }
          status_t status = matmul_direct(
                              'r',  // layout: row-major
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, bias_data,
                              beta, C_data, ldc, true,
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
      if (skip) {
        continue;
      }

      // Store total time for each layer
      for (size_t i = 0; i < cfg.n_values.size(); i++) {
        time_stats_layer[i].total_time_ms = elapsed_ms_layer[i];
      }
      print_matmul_execution_summary(cfg, time_stats_layer, options);
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