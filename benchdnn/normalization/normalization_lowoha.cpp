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

#include "normalization_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace normalization {

int normalization_lowoha_benchdnn(
  std::vector<NormalizationConfig> configs,
  std::vector<std::pair<NormalizationConfig, TimingStats>> &normalization_results,
  size_t cache_size) {

  bool skip;
  for (const auto &cfg : configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t input_tensor, output_tensor, gamma_tensor, beta_tensor,
               running_mean_tensor, running_var_tensor, residual_tensor;

      int ret = create_input_tensor(tensor_factory, cfg, input_tensor);
      if (ret != OK) {
        testlog_error("create_input_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      if (!cfg.isInplace) {
        ret = create_output_tensor(tensor_factory, cfg, output_tensor);
        if (ret != OK) {
          testlog_error("create_output_tensor failed");
          log_benchmark_failure(cfg);
          continue;
        }
      }

      ret = create_gamma_tensor(tensor_factory, cfg, gamma_tensor);
      if (ret != OK) {
        testlog_error("create_gamma_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_beta_tensor(tensor_factory, cfg, beta_tensor);
      if (ret != OK) {
        testlog_error("create_beta_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_running_mean_tensor(tensor_factory, cfg,
                                       running_mean_tensor);
      if (ret != OK) {
        testlog_error("create_running_mean_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_running_var_tensor(tensor_factory, cfg,
                                      running_var_tensor);
      if (ret != OK) {
        testlog_error("create_running_var_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_residual_tensor(tensor_factory, cfg, residual_tensor);
      if (ret != OK) {
        testlog_error("create_residual_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      // Get raw pointers for LOWOHA direct API
      void *input_data = input_tensor.get_raw_handle_unsafe();
      void *output_data = cfg.isInplace ?
                          input_data : output_tensor.get_raw_handle_unsafe();
      const void *gamma_data = cfg.use_scale ?
                               gamma_tensor.get_raw_handle_const() : nullptr;
      const void *beta_data = (cfg.use_shift &&
                               cfg.norm_type != "rms_norm" &&
                               cfg.norm_type != "fused_add_rms_norm") ?
                              beta_tensor.get_raw_handle_const() : nullptr;
      const void *running_mean_data = (cfg.norm_type == "batch_norm") ?
                                      running_mean_tensor.get_raw_handle_const() : nullptr;
      const void *running_var_data = (cfg.norm_type == "batch_norm") ?
                                     running_var_tensor.get_raw_handle_const() : nullptr;
      void *residual_data = (cfg.norm_type == "fused_add_rms_norm") ?
                            residual_tensor.get_raw_handle_unsafe() : nullptr;

      // Setup LOWOHA normalization parameters
      norm_params params;
      params.norm_type    = strToLowohaType(cfg.norm_type);
      params.batch        = cfg.batch;
      params.norm_size    = cfg.norm_size;
      params.num_channels = cfg.num_channels;
      params.src_dt = cfg.src_dt;
      params.dst_dt = cfg.dst_dt;
      params.gamma_dt = cfg.gamma_dt;
      params.beta_dt = cfg.beta_dt;
      params.epsilon = cfg.epsilon;
      params.use_scale = cfg.use_scale;
      params.use_shift = cfg.use_shift;
      params.algorithm = strToLowohaAlgo(cfg.algorithm);
      params.num_threads = cfg.num_threads;

      if (params.norm_type == norm_type_t::NONE) {
        testlog_error("LOWOHA: Unknown norm_type: ", cfg.norm_type);
        log_benchmark_failure(cfg);
        continue;
      }

      TimingStats time_stats;

      // Warm-up iterations
      for (int i = 0; i < cfg.warmup_iters && !skip; ++i) {
        status_t status = normalization_direct(
                            input_data, output_data, gamma_data, beta_data,
                            running_mean_data, running_var_data,
                            residual_data, params);
        if (status != status_t::success) {
          testlog_error("LOWOHA: Normalization execution failed during warm-up.");
          skip = true;
          break;
        }
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      double elapsed_ms = 0.0;

      // Benchmark iterations
      for (int i = 0; i < cfg.iters && !skip; ++i) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        auto start = std::chrono::high_resolution_clock::now();

        status_t status = normalization_direct(
                            input_data, output_data, gamma_data, beta_data,
                            running_mean_data, running_var_data,
                            residual_data, params);

        if (status != status_t::success) {
          testlog_error("LOWOHA: Normalization execution failed during benchmark iterations.");
          skip = true;
          break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      time_stats.total_time_ms = elapsed_ms;
      normalization_results.emplace_back(cfg, time_stats);

    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  return OK;
}

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl
