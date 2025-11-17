/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// - Add bias support.
// - Add post-op support.
// - Add pipeline (multi-layer) support.
int matmul_lowoha_benchdnn(std::vector<MatmulConfig> configs,
                           std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                           &matmul_results, const global_options &options) {

  bool skip;
  for (const auto &cfg:configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t input_tensor;
      std::vector<tensor_t> weight_tensor, bias, output_tensor;

      int ret = create_weights_tensor(tensor_factory, cfg, weight_tensor, options);
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

      float alpha = 1.0f, beta = 0.0f;
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

      data_types matmul_dtypes;
      matmul_dtypes.src = cfg.dt[0];
      matmul_dtypes.wei = cfg.dt[1];
      matmul_dtypes.dst = cfg.dt[2];
      matmul_dtypes.bias = cfg.bias_dt;
      matmul_dtypes.compute = data_type_t::none;

      lowoha_params params;
      params.dtypes = matmul_dtypes;

      batch_params_t batch_params;
      batch_params.Batch_A = batchA;
      batch_params.Batch_B = batchB;

      // Validate data types
      if (cfg.dt[0] != data_type_t::f32 && cfg.dt[0] != data_type_t::bf16) {
        testlog_error("LOWOHA: Unsupported source data type");
        log_benchmark_failure(cfg);
        continue;
      }
      if (cfg.dt[1] != data_type_t::f32 && cfg.dt[1] != data_type_t::bf16) {
        testlog_error("LOWOHA: Unsupported weight data type");
        log_benchmark_failure(cfg);
        continue;
      }
      if (cfg.dt[2] != data_type_t::f32 && cfg.dt[2] != data_type_t::bf16) {
        testlog_error("LOWOHA: Unsupported output data type");
        log_benchmark_failure(cfg);
        continue;
      }

      TimingStats time_stats;
      // warm-up iterations
      for (auto j = 0; j < cfg.warmup_iters && !skip; j++) {
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          void *A_data = (i == 0) ? input_tensor.get_raw_handle_unsafe() :
                         output_tensor[i - 1].get_raw_handle_unsafe();
          void *B_data = weight_tensor[i].get_raw_handle_unsafe();
          void *C_data = output_tensor[i].get_raw_handle_unsafe();

          status_t status = matmul_direct(
                              'r',  // layout: row-major
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, nullptr,  // No bias
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
        std::vector<char> buffer(CACHE_SIZE, 1);
        flush_cache(buffer);
#endif
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          auto start_layer = std::chrono::high_resolution_clock::now();
          TimingStats time_stats; // Per-layer, per-iteration

          void *A_data = (i == 0) ? input_tensor.get_raw_handle_unsafe() :
                         output_tensor[i - 1].get_raw_handle_unsafe();
          void *B_data = weight_tensor[i].get_raw_handle_unsafe();
          void *C_data = output_tensor[i].get_raw_handle_unsafe();
          status_t status = matmul_direct(
                              'r',  // layout: row-major
                              cfg.isTransA, cfg.isTransB,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              alpha, A_data, lda, B_data, ldb, nullptr,  // No bias
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